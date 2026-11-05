from __future__ import annotations

import torch.nn.functional as F

from model.coevo_st import EvoCoPlugin
from model.strap import STRAP_Model

class _Batch:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

class CoEvo_STRAP_Model(EvoCoPlugin, STRAP_Model):
    def __init__(self, args):
        STRAP_Model.__init__(self, args)
        self._init_coevo(args, int(args.gcn["out_channel"]))

    def _trunk(self, x, adj):
        year = getattr(self.args, "year", None)
        if year is not None:
            self.current_year = year
        adj = self._prepare_strap(_Batch(x), adj)
        enhanced = self._apply_strap(self.backbone(x, adj))
        out_ch = int(self.args.gcn["out_channel"])
        if enhanced.shape[-1] != out_ch:
            b, n, c = enhanced.shape
            enhanced = F.adaptive_avg_pool1d(enhanced.reshape(-1, c).unsqueeze(1), out_ch).squeeze(1)
            enhanced = enhanced.reshape(b, n, out_ch)
        return enhanced + x

    def expand_adaptive_params(self, new_num_nodes):
        pass
