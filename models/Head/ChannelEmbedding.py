from torch import nn
from torch.nn import functional as F
from .utils import ChannelPool


class ChannelEmbedding(nn.Module):
    def __init__(self, input_ndim, mid_process_channels, **kwargs):
        super(ChannelEmbedding, self).__init__()
        last_ndim = input_ndim
        self.mid_process_nums = len(mid_process_channels)
        for module_index, channel in enumerate(mid_process_channels):
            setattr(self, "module_{}".format(module_index), nn.Linear(last_ndim, channel))
            last_ndim = channel
    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, z, x):
        # zb, zc, zh, zw = z.shape
        xb, xc, xh, xw = x.shape
        x = x.contiguous().view(xb, xc, xh*xw).transpose(1, 2).contiguous()
        for ind in range(self.mid_process_nums):
            x = getattr(self, "module_{}".format(ind))(x)
        x = x.transpose(1, 2).view(xb, 1, xh, xw).contiguous()
        return x, None


class ChannelEmbeddingOffset(nn.Module):
    def __init__(self, input_ndim, mid_process_channels, reg_mid_process_channels, reg_scale=1.0, **kwargs):
        super(ChannelEmbeddingOffset, self).__init__()
        self.cls_head = ChannelEmbedding(input_ndim, mid_process_channels)
        self.reg_scale = float(reg_scale)
        last_ndim = input_ndim
        self.reg_process_nums = len(reg_mid_process_channels)
        for module_index, channel in enumerate(reg_mid_process_channels):
            layer = nn.Linear(last_ndim, channel)
            setattr(self, "reg_module_{}".format(module_index), layer)
            last_ndim = channel
        if self.reg_process_nums > 0:
            last_layer = getattr(self, "reg_module_{}".format(self.reg_process_nums - 1))
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, z, x):
        cls, _ = self.cls_head(z, x)
        xb, xc, xh, xw = x.shape
        reg = x.contiguous().view(xb, xc, xh * xw).transpose(1, 2).contiguous()
        for ind in range(self.reg_process_nums):
            reg = getattr(self, "reg_module_{}".format(ind))(reg)
        reg = reg.transpose(1, 2).view(xb, 2, xh, xw).contiguous()
        return cls, reg * self.reg_scale
