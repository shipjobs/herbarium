# Copyright (c) Facebook, Inc. and its affiliates.
from collections import defaultdict
from herbarium.utils.comm import all_gather
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor, nn
from torch.nn import functional as F

from herbarium.config import configurable
from herbarium.data.dataset_utils import convert_image_to_rgb
from herbarium.layers import ShapeSpec, cat, get_norm, nonzero_tuple, AttentionConv
from herbarium.structures import ImageList, Instances
from herbarium.utils.events import get_event_storage
from herbarium.data.catalog import DatasetCatalog, MetadataCatalog
from sklearn.metrics import f1_score

from ..backbone import build_backbone
from .build import META_ARCH_REGISTRY

__all__ = ["SimpleNet"]


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class SimpleNet(nn.Module):
    """
    Implement RetinaNet in :paper:`SimpleNet`.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        head,
        head_in_features,
        num_classes,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
        hierarchy_loss=False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow herbarium's backbone interface
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            num_classes (int): number of classes. Used to label background proposals.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            smooth_l1_beta (float): smooth_l1_beta
            box_reg_loss_type (str): Options are "smooth_l1", "giou"

            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).

            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
            vis_period (int):
                The period (in terms of steps) for minibatch visualization at train time.
                Set to 0 to disable.
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features

        self.num_classes = num_classes
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format
        self.hierarchy_loss = hierarchy_loss

        self.cls_loss_func = nn.CrossEntropyLoss()
        self.div_loss_func = nn.KLDivLoss()
        

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape() 
        #final_channel = backbone._out_feature_channels["blockE"]
        feature_shapes = {f:backbone_shape[f] for f in cfg.MODEL.SIMPLENET.IN_FEATURES}
        head = SimpleNetHead(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.SIMPLENET.NUM_CLASSES,
            "head_in_features": cfg.MODEL.SIMPLENET.IN_FEATURES,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
            "hierarchy_loss": cfg.MODEL.SIMPLENET.HIERARCHY_LOSS,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def update_classifier(self):
        self.head.update_hierarchy_classifier()

    def update_prior(self):
        self.head.update_hierarchy_prior()

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
            in inference, the standard output format, described in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = {key: features[key] for key in features.keys() if key in self.head_in_features}

        pred_logits = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        #pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "annotations" in batched_inputs[0], "annotations are missing in training!"
            gt_annotations = [x["annotations"] for x in batched_inputs]

            losses = self.losses(pred_logits, gt_annotations)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        pred_logits, images.image_sizes
                    )
                    #self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(pred_logits, images.image_sizes)
            return results

    def losses(self, pred_logits, gt_labels):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)

        if self.hierarchy_loss:
            pass
        else:
            species_labels = torch.tensor([labels[0]["category_id"] for labels in gt_labels]).cuda()
            family_labels, order_labels = self.head.label_from_prior(species_labels)

            species_loss = self.cls_loss_func(pred_logits["species"], species_labels)
            family_loss = self.div_loss_func(F.log_softmax(pred_logits["family"], dim=1), family_labels)
            order_loss = self.div_loss_func(F.log_softmax(pred_logits["order"], dim=1), order_labels)

            loss = {
                "species_loss": species_loss,
                "family_loss": family_loss,
                "order_loss": order_loss,
            }

        return loss

    def visualize_training(self, batched_inputs, results):
        #from herbarium.utils.visualizer import Visualizer
        assert len(batched_inputs) == len(results), "Cannot visualize inputs and results of different sizes"

        storage = get_event_storage()
        y_true = batched_inputs["annotations"]
        #v_gt = Visualizer(gt_annotation, results)
        vis_name = f"per-batch f1 score"
        storage.put_scalar(vis_name, f1_score(y_true, results))

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def inference(
        self,
        pred_logits: Dict[str, Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        num_images = pred_logits["leaf"].shape[0]
        pred = pred_logits["leaf"].view(num_images, -1)
        result = pred.view(num_images, -1).argmax(dim=1)
        return result

    def inference_single_image(
        self,
        anchors: List[Tensor],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class SimpleNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: Dict[str,ShapeSpec],
        num_classes: Dict[str, int],
        norm="",
        prior_prob=0.01,
        final_channel=512,
        hierarchy_loss=False,
        hierarchy_prior=None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`herbarium.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__()

        if norm == "BN" or norm == "SyncBN":
            logger = logging.getLogger(__name__)
            logger.warn("Shared norm does not work well for BN, SyncBN, expect poor results")

        cls_subnet = defaultdict(list)
        for input_key in list(input_shape.keys()):

            in_channels = input_shape[input_key].channels
            out_channels = final_channel

            cls_subnet[input_key] = [AttentionConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

            if norm:
                cls_subnet[input_key].append(get_norm(norm, out_channels))
            cls_subnet[input_key].append(nn.ReLU())
            cls_subnet[input_key].append(nn.AdaptiveAvgPool2d((1,1)))
            cls_subnet[input_key] = nn.Sequential(*cls_subnet[input_key])

        self.cls_subnet = nn.ModuleDict(cls_subnet)

        cls_score = defaultdict(list)
        for cls_level in list(num_classes.keys()):

            cls_score[cls_level] = nn.Conv2d(
                final_channel, num_classes[cls_level], kernel_size=1, stride=1
            )

        self.cls_score = nn.ModuleDict(cls_score)
        self.hierarchy_prior = nn.ParameterDict({
            k: nn.Parameter(v, requires_grad=False) for k, v in hierarchy_prior.items()
        })

        # Initialization
        for modules in [self.cls_subnet, self.cls_score]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    if layer.weight is not None:
                        torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

        # Initialize higher class's classifier with lower class weight / bias and hierarchy priority

        #self.update_hierarchy_classifier()

        # Use prior in model initialization to improve stability
        #bias_value = -(math.log((1 - prior_prob) / prior_prob))
        #torch.nn.init.constant_(self.cls_score.bias, bias_value)
    
    def label_from_prior(self, species_label):
        family_label = self.hierarchy_prior["family|species"][species_label]
        order_label = torch.einsum('ik,kj->ij',family_label,self.hierarchy_prior["order|family"])
        return family_label, order_label 

    def update_hierarchy_prior(self):
        species_classifier = self.cls_score["species"]
        family_classifier = self.cls_score["family"]
        order_classifier = self.cls_score["order"]       

        

        pass
    
    def update_hierarchy_classifier(self):
        species_classifier = self.cls_score["species"]
        family_classifier = self.cls_score["family"]
        order_classifier = self.cls_score["order"]

        family_species_hierarchy = self.hierarchy_prior["family|species"].t()
        order_family_hierarchy = self.hierarchy_prior["order|family"].t()
        order_species_hierarchy = torch.einsum("ik,kj->ij", order_family_hierarchy, family_species_hierarchy)

        F, S = family_species_hierarchy.shape
        O, S = order_species_hierarchy.shape
        C = species_classifier.weight.shape[1]

        species_weight = species_classifier.state_dict()["weight"].view(S, C)
        species_bias = species_classifier.state_dict()["bias"]
        family_classifier_state = family_classifier.state_dict()
        order_classifier_state = order_classifier.state_dict()

        family_classifier_state["weight"] = torch.einsum('ik,kj->ij',family_species_hierarchy,species_weight).view(F, C, 1, 1)
        order_classifier_state["weight"] = torch.einsum('ik,kj->ij',order_species_hierarchy,species_weight).view(O, C, 1, 1)

        family_classifier_state["bias"] = torch.einsum('ij,j->i',family_species_hierarchy,species_bias)
        order_classifier_state["bias"] = torch.einsum('ij,j->i',order_species_hierarchy,species_bias)

        family_classifier.load_state_dict(family_classifier_state)
        order_classifier.load_state_dict(order_classifier_state)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec], final_channel = 512):
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        num_classes = {f: meta.num_classes[f] for f in cfg.MODEL.SIMPLENET.NUM_CLASSES}
        hierarchy_prior = {
            "order|family": meta.order_family_hierarchy,
            "family|species": meta.family_species_hierarchy,
        }
        return {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "prior_prob": cfg.MODEL.SIMPLENET.PRIOR_PROB,
            "norm": cfg.MODEL.SIMPLENET.NORM,
            "final_channel": final_channel,
            "hierarchy_loss": cfg.MODEL.SIMPLENET.HIERARCHY_LOSS,
            "hierarchy_prior": hierarchy_prior
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        attention_feature = 0
        logits = {}

        for key, feature in features.items():
            attention_feature += self.cls_subnet[key](feature)

        N = attention_feature.shape[0]

        for key, pred_cls in self.cls_score.items():
            logits[key] = pred_cls(attention_feature).view(N, -1)

        return logits
