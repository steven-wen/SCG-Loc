import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualContextMixer(nn.Module):
    def __init__(self, channels, hidden_ratio=4):
        super().__init__()
        hidden_dim = max(channels // max(hidden_ratio, 1), 32)

        self.dw3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.dw3_d2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
        )

        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x):
        local = self.dw3(x)
        dilated = self.dw3_d2(x)
        global_context = self.global_proj(x).expand_as(x)
        delta = self.fuse(torch.cat([x, local, dilated, global_context], dim=1))
        return x + delta


class AdaptiveTopDownGate(nn.Module):
    def __init__(self, channels, hidden_ratio=4, gate_bias=2.0):
        super().__init__()
        hidden_dim = max(channels // max(hidden_ratio, 1), 32)

        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.channel_gate = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)
        self.spatial_gate = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)

        nn.init.zeros_(self.channel_gate.weight)
        nn.init.constant_(self.channel_gate.bias, gate_bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.constant_(self.spatial_gate.bias, gate_bias)

    def forward(self, lateral, topdown):
        if topdown.shape[-2:] != lateral.shape[-2:]:
            topdown = F.interpolate(topdown, size=lateral.shape[-2:], mode="nearest")

        interaction = torch.cat(
            [
                lateral,
                topdown,
                (lateral - topdown).abs(),
                lateral * topdown,
            ],
            dim=1,
        )
        hidden = self.encoder(interaction)
        gate = torch.sigmoid(self.channel_gate(hidden)) * torch.sigmoid(self.spatial_gate(hidden))
        return lateral + gate * topdown


class DetailResidualExtractor(nn.Module):
    def __init__(self, channels, hidden_ratio=4):
        super().__init__()
        hidden_dim = max(channels // max(hidden_ratio, 1), 32)

        self.dw3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.dw5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False,
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
        )
        self.detail_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)
        nn.init.zeros_(self.detail_gate[1].weight)
        nn.init.zeros_(self.detail_gate[1].bias)

    def forward(self, x):
        detail = torch.cat([self.dw3(x), self.dw5(x)], dim=1)
        detail = self.project(detail)
        return self.detail_gate(x) * detail


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return x + self.block(x)


class DetailPreservingContextGatedFPN_I3(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims=128,
        context_hidden_ratio=4,
        fusion_hidden_ratio=4,
        detail_hidden_ratio=4,
        gate_bias=2.0,
        **kwargs,
    ):
        super().__init__()
        c3_size, c4_size, c5_size = input_dims

        self.l3 = nn.Conv2d(c3_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.l4 = nn.Conv2d(c4_size, output_dims, kernel_size=1, stride=1, padding=0)
        self.l5 = nn.Conv2d(c5_size, output_dims, kernel_size=1, stride=1, padding=0)

        self.c5_context = ResidualContextMixer(
            output_dims,
            hidden_ratio=context_hidden_ratio,
        )
        self.fuse_p4 = AdaptiveTopDownGate(
            output_dims,
            hidden_ratio=fusion_hidden_ratio,
            gate_bias=gate_bias,
        )
        self.fuse_p3 = AdaptiveTopDownGate(
            output_dims,
            hidden_ratio=fusion_hidden_ratio,
            gate_bias=gate_bias,
        )
        self.p3_detail = DetailResidualExtractor(
            output_dims,
            hidden_ratio=detail_hidden_ratio,
        )

        self.p5_out = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1)
        self.p4_out = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1)
        self.p3_out = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1)
        self.p3_refine = ResidualRefineBlock(output_dims)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        l3 = self.l3(c3)
        l4 = self.l4(c4)
        l5 = self.l5(c5)

        p5_inner = self.c5_context(l5)
        p5 = self.p5_out(p5_inner)

        p4_inner = self.fuse_p4(l4, p5_inner)
        p4 = self.p4_out(p4_inner)

        p3_inner = self.fuse_p3(l3, p4)
        p3_inner = p3_inner + self.p3_detail(l3)
        p3 = self.p3_out(p3_inner)
        p3 = self.p3_refine(p3)

        return [p3, p4, p5]
