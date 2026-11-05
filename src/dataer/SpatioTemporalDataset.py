import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

class SpatioTemporalDataset(TorchDataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x']
            self.y = inputs[split+'_y']
        else:
            self.x = x
            self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        return Data(x=x, y=y)

class continue_learning_Dataset(TorchDataset):
    def __init__(self, inputs):
        self.x = inputs

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return Data(x=x)

class CaqraMultimodalDataset(SpatioTemporalDataset):

    def __init__(self, inputs, split, args, year: int):
        super().__init__(inputs, split)
        self.args = args
        self.year = int(year)
        key = f"{split}_t_end"
        if key not in inputs:
            raise KeyError(f"CaqraMultimodalDataset requires {key} in inputs (enable multimodal_caqra + data_process).")
        self.t_end = inputs[key]
        res = str(getattr(args, "temporal_resolution", "hourly") or "hourly")
        self.hours_per_step = 24 if res == "daily" else int(getattr(args, "hours_per_step", 1) or 1)
        self.minutes_per_step = float(getattr(args, "minutes_per_step", 0) or 0)
        from utils.caqra_io import image_mode, timeline_to_datetime, context_datetimes, warm_caqra_cache

        self._mode = image_mode(args)
        self._t2dt = timeline_to_datetime
        self._ctx_dts = context_datetimes
        warm_caqra_cache(args, self.year, self.t_end)

    def _context(self, te: int):
        if self.minutes_per_step > 0:
            minutes = float(te) * self.minutes_per_step
            hod = (minutes / 60.0) % 24.0
            doy = (minutes / (60.0 * 24.0)) % 366.0
        else:
            hours = int(te) * self.hours_per_step
            hod = hours % 24
            doy = (hours // 24) % 366
        two_pi = 2.0 * np.pi
        return torch.tensor(
            [
                np.sin(two_pi * hod / 24.0), np.cos(two_pi * hod / 24.0),
                np.sin(two_pi * doy / 366.0), np.cos(two_pi * doy / 366.0),
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, index):
        from utils.caqra_io import load_caqra_stack

        data = super().__getitem__(index)
        te = int(self.t_end[index])
        data.img = load_caqra_stack(self.args, self.year, te, device=None)
        data.ctx = self._context(te)
        if self.minutes_per_step > 0:
            stride = max(1, int(getattr(self.args, "caqra_frame_stride", 3)))
            data.age = torch.tensor([stride * self.minutes_per_step / 60.0], dtype=torch.float32)
        else:
            origin = self._t2dt(self.args, self.year, te)
            newest = self._ctx_dts(self.args, self.year, te)[0]
            data.age = torch.tensor([(origin - newest).total_seconds() / 3600.0], dtype=torch.float32)
        data.t_end = torch.tensor([te], dtype=torch.long)
        return data
