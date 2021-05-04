# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# File:


from . import catalog as _UNUSED  # register the handler
#from .detection_checkpoint import DetectionCheckpointer
# TODO: We don't use detection_checkpointer but might have to make checkpointer based on this
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

__all__ = ["Checkpointer", "PeriodicCheckpointer", "DetectionCheckpointer"]
