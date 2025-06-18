import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from .module import MetaModule

class MetaLayerNorm(nn.LayerNorm, MetaModule):
    __doc__ = nn.LayerNorm.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)


class MetaGroupNorm(nn.GroupNorm, MetaModule):
    __doc__ = nn.GroupNorm.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.group_norm(
            input, self.num_groups, weight, bias, self.eps)
