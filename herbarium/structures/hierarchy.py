# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import List, Tuple, Union
import torch
from torch import device

from herbarium.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Hierarchy:
    """
    This structure stores a hierarchy as a Nx3 (order, family, species) torch.Tensor.
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all id)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx3. Each row is (order, family, species).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx3 matrix.  Each row is (order, family, species).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Hierarchy":
        """
        Clone the Hierarchy.

        Returns:
            Hierarchy
        """
        return Hierarchy(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device):
        # Hierarchy are assumed float32 and does not support to(dtype)
        return Hierarchy(self.tensor.to(device=device))

    def __getitem__(self, item) -> "Hierarchy":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Hierarchy: Create a new :class:`Hierarchy` by indexing.

        The following usage are allowed:

        1. `new_hi = hierarchy[3]`: return a `Hierarchy` which contains only one box.
        2. `new_hi = hierarchy[2:10]`: return a slice of hierarchy.
        3. `new_hi = hierarchy[vector]`, where vector is a torch.BoolTensor
           with `length = len(hierarchy)`. Nonzero elements in the vector will be selected.

        Note that the returned Hierarchy might share storage with this hierarchy,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Hierarchy(self.tensor[item].view(1, -1))
        h = self.tensor[item]
        assert h.dim() == 2, "Indexing on Hierarchy with {} failed to return a matrix!".format(item)
        return Hierarchy(h)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Hierarchy(" + str(self.tensor) + ")"

    @classmethod
    @_maybe_jit_unused
    def cat(cls, hierarchy_list: List["Hierarchy"]) -> "Hierarchy":
        """
        Concatenates a list of Hierarchy into a single Hierarchy 

        Arguments:
            hierarcy_list (list[Hierarchy])

        Returns:
            Hierarchy: the concatenated Hierarchy 
        """
        assert isinstance(hierarchy_list, (list, tuple))
        if len(hierarchy_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(hierarchy, Hierarchy) for hierarchy in hierarchy_list])

        # use torch.cat (v.s. layers.cat) so the returned hierarchy never share storage with input
        cat_hierarchy = cls(torch.cat([h.tensor for h in hierarchy_list], dim=0))
        return cat_hierarchy

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (3,) at a time.
        """
        yield from self.tensor