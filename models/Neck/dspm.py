import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels, norm_groups):
    groups = min(int(norm_groups), int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvGNGELU(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, norm_groups=32):
        super().__init__()
        out_channels = int(out_channels or in_channels)
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_valid_groups(out_channels, norm_groups), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DSPM(nn.Module):
    """Dynamic Structural Prototype Matching.

    DSPM builds a structural prototype from the UAV feature pyramid, matches
    it against local satellite structures, and injects the matched context
    into high-resolution satellite features.
    """

    def __init__(
        self,
        channels=128,
        match_dim=64,
        token_grid=3,
        hidden_channels=128,
        norm_groups=32,
        temperature=0.1,
        dual_softmax=True,
        gate_bias=-1.0,
        alpha_init=0.0,
        center_prior=False,
        center_prior_min=0.5,
        center_prior_power=2.0,
        center_prior_on="value",
        global_query_score=False,
        global_query_score_weight=0.5,
        global_query_pool_alpha=0.1,
        p3_update_scale=1.0,
        update_p4=False,
        p4_update_scale=0.5,
    ):
        super().__init__()
        token_grid = int(token_grid)
        if token_grid < 1 or token_grid % 2 == 0:
            raise ValueError("token_grid must be a positive odd integer.")

        self.channels = int(channels)
        self.match_dim = int(match_dim)
        self.token_grid = token_grid
        self.num_tokens = token_grid * token_grid
        self.temperature = max(float(temperature), 1e-6)
        self.dual_softmax = bool(dual_softmax)
        self.center_prior_enabled = bool(center_prior)
        self.center_prior_on = str(center_prior_on).lower()
        if self.center_prior_on not in ("value", "logits", "both"):
            raise ValueError("center_prior_on must be one of: value, logits, both.")
        self.global_query_score_enabled = bool(global_query_score)
        self.global_query_score_weight = max(float(global_query_score_weight), 0.0)
        self.global_query_pool_alpha = float(global_query_pool_alpha)
        self.p3_update_scale = float(p3_update_scale)
        self.update_p4 = bool(update_p4)
        self.p4_update_scale = float(p4_update_scale)

        center_prior_tensor = self._build_center_prior(
            token_grid,
            min_weight=float(center_prior_min),
            power=float(center_prior_power),
        )
        self.register_buffer(
            "center_prior",
            center_prior_tensor.view(1, self.num_tokens, 1),
            persistent=False,
        )

        self.uav_key = nn.Conv2d(channels, match_dim, kernel_size=1, bias=False)
        self.sat_key = nn.Conv2d(channels, match_dim, kernel_size=1, bias=False)
        self.uav_value = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        coord = torch.linspace(-1.0, 1.0, steps=token_grid)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        rel = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        self.register_buffer("relative_coords", rel, persistent=False)
        self.relative_mlp = nn.Sequential(
            nn.Linear(2, match_dim),
            nn.GELU(),
            nn.Linear(match_dim, match_dim),
        )

        hidden_channels = int(hidden_channels)
        self.context_refine = nn.Sequential(
            ConvGNGELU(
                channels * 4 + 1,
                hidden_channels,
                kernel_size=1,
                norm_groups=norm_groups,
            ),
            ConvGNGELU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                norm_groups=norm_groups,
            ),
            ConvGNGELU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.context_project = nn.Sequential(
            ConvGNGELU(
                channels,
                hidden_channels,
                kernel_size=3,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.gate_project = nn.Sequential(
            ConvGNGELU(
                channels * 2 + 1,
                hidden_channels,
                kernel_size=1,
                norm_groups=norm_groups,
            ),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.gate_project[-1].bias, float(gate_bias))

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, sat_features, uav_features):
        if len(sat_features) < 3 or len(uav_features) < 3:
            raise ValueError("DSPM expects [P3, P4, P5] for SAT and UAV.")

        sat_p3, sat_p4, sat_p5 = sat_features[:3]
        uav_p5 = uav_features[2]
        batch_size, _, sat_h5, sat_w5 = sat_p5.shape

        uav_key_map = self.uav_key(uav_p5)
        sat_key_map = self.sat_key(sat_p5)
        uav_key_tokens = self._uav_tokens(uav_key_map)
        uav_value_tokens = self._uav_tokens(
            self.uav_value(uav_p5),
            normalize=False,
        )
        sat_key_tokens = self._satellite_tokens(sat_key_map)

        if self.center_prior_enabled and self.center_prior_on in ("value", "both"):
            uav_value_tokens = uav_value_tokens * self.center_prior.to(
                device=uav_value_tokens.device,
                dtype=uav_value_tokens.dtype,
            )

        pos = self.relative_mlp(
            self.relative_coords.to(device=sat_p5.device, dtype=sat_p5.dtype)
        )
        uav_key_tokens = F.normalize(uav_key_tokens + pos.unsqueeze(0), dim=-1)
        sat_key_tokens = F.normalize(
            sat_key_tokens
            + pos.view(1, 1, self.num_tokens, self.match_dim),
            dim=-1,
        )

        logits = (
            torch.einsum("blnd,bmd->blnm", sat_key_tokens, uav_key_tokens)
            / self.temperature
        )
        if self.center_prior_enabled and self.center_prior_on in ("logits", "both"):
            log_prior = (
                self.center_prior.squeeze(-1)
                .clamp_min(1e-6)
                .log()
                .to(device=logits.device, dtype=logits.dtype)
            )
            logits = logits + log_prior.view(1, 1, 1, self.num_tokens)

        if self.dual_softmax:
            attn = F.softmax(logits, dim=-1) * F.softmax(logits, dim=-2)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            score = attn.amax(dim=-1).mean(dim=-1)
        else:
            attn = F.softmax(logits, dim=-1)
            score = attn.amax(dim=-1).mean(dim=-1)

        if self.global_query_score_enabled and self.global_query_score_weight > 0:
            query_score = self._query_global_score(uav_key_map, sat_key_map)
            weight = self.global_query_score_weight
            score = (score + weight * query_score) / (1.0 + weight)

        matched_tokens = torch.einsum("blnm,bmc->blnc", attn, uav_value_tokens)
        matched_context = (
            matched_tokens.mean(dim=2)
            .transpose(1, 2)
            .reshape(batch_size, self.channels, sat_h5, sat_w5)
        )
        score_map = score.reshape(batch_size, 1, sat_h5, sat_w5)

        p5_context = self.context_refine(
            torch.cat(
                [
                    sat_p5,
                    matched_context,
                    (sat_p5 - matched_context).abs(),
                    sat_p5 * matched_context,
                    score_map,
                ],
                dim=1,
            )
        )
        context3 = F.interpolate(
            p5_context,
            size=sat_p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        score3 = F.interpolate(
            score_map,
            size=sat_p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        context_feat = self.context_project(context3)
        gate = torch.sigmoid(
            self.gate_project(torch.cat([sat_p3, context3, score3], dim=1))
        )
        p3_final = (
            sat_p3
            + self.p3_update_scale * self.alpha * gate * context_feat
        )

        output = list(sat_features)
        output[0] = p3_final
        if self.update_p4:
            context4 = F.interpolate(
                p5_context,
                size=sat_p4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            score4 = F.interpolate(
                score_map,
                size=sat_p4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            context_feat4 = self.context_project(context4)
            gate4 = torch.sigmoid(
                self.gate_project(torch.cat([sat_p4, context4, score4], dim=1))
            )
            output[1] = (
                sat_p4
                + self.p4_update_scale * self.alpha * gate4 * context_feat4
            )
        return output

    @staticmethod
    def _build_center_prior(token_grid, min_weight=0.5, power=2.0):
        min_weight = min(max(float(min_weight), 0.0), 1.0)
        power = max(float(power), 1e-6)
        coord = torch.arange(token_grid, dtype=torch.float32)
        center = (float(token_grid) - 1.0) / 2.0
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        dist = torch.sqrt((yy - center).pow(2) + (xx - center).pow(2))
        max_dist = torch.sqrt(
            torch.tensor(2.0 * center * center, dtype=torch.float32)
        ).clamp_min(1e-6)
        prior = (1.0 - dist / max_dist).clamp_min(0.0).pow(power)
        prior = min_weight + (1.0 - min_weight) * prior
        return prior.reshape(-1)

    def _uav_tokens(self, feature, normalize=True):
        pooled = F.adaptive_avg_pool2d(
            feature,
            (self.token_grid, self.token_grid),
        )
        tokens = pooled.flatten(2).transpose(1, 2)
        if normalize:
            tokens = F.normalize(tokens, dim=-1)
        return tokens

    def _satellite_tokens(self, feature):
        batch_size, channels, height, width = feature.shape
        unfolded = F.unfold(
            feature,
            kernel_size=self.token_grid,
            padding=self.token_grid // 2,
        )
        tokens = unfolded.transpose(1, 2).reshape(
            batch_size,
            height * width,
            self.num_tokens,
            channels,
        )
        return tokens

    def _query_global_score(self, uav_key_map, sat_key_map):
        q_avg = F.adaptive_avg_pool2d(uav_key_map, 1).flatten(1)
        q_max = F.adaptive_max_pool2d(uav_key_map, 1).flatten(1)
        query = F.normalize(
            q_avg + self.global_query_pool_alpha * q_max,
            dim=1,
        )
        sat_tokens = F.normalize(
            sat_key_map.flatten(2).transpose(1, 2),
            dim=-1,
        )
        score = torch.einsum("bld,bd->bl", sat_tokens, query)
        score_min = score.amin(dim=1, keepdim=True)
        score_max = score.amax(dim=1, keepdim=True)
        return (score - score_min) / (score_max - score_min).clamp_min(1e-6)
