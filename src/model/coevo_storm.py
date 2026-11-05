from __future__ import annotations

from model.coevo_st import EvoCoPlugin
from model.storm import Storm_Model

class CoEvo_Storm_Model(EvoCoPlugin, Storm_Model):
    def __init__(self, args):
        Storm_Model.__init__(self, args)
        self._init_coevo(args, int(self.channels[1][2]))

    def _trunk(self, x, adj):
        self._sync_graph(adj)
        return self.net.encode(x.permute(0, 2, 1).unsqueeze(-1).contiguous())
