from __future__ import annotations

import torch.nn.functional as F

from model.coevo_st import EvoCoPlugin
from model.stbp import STBP_Model

class CoEvo_STBP_Model(EvoCoPlugin, STBP_Model):
    FREEZE_PREFIXES = "fconv1,stmodule,fconv2,fc."

    def __init__(self, args):
        STBP_Model.__init__(self, args)
        self._init_coevo(args, int(args.model["out_channel"]))

    def _trunk(self, x, adj):
        b, n = x.shape[0], x.shape[1]
        h = F.relu(self.fconv1(x))
        node_emb = self.pattern_bank[:n].unsqueeze(0).expand(b, n, -1).permute(0, 2, 1).unsqueeze(-1)
        h = self.stmodule(h.permute(0, 2, 1).unsqueeze(-1), node_emb).squeeze(-1)
        h = self.fconv2(h.permute(0, 2, 1).squeeze(-1))
        return h + x
