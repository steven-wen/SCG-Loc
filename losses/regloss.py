from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from .utils import create_labels, create_labels_2


class SmoothL1Loss(nn.Module):
    def __init__(self, center_R, weight_rate=1.0, score_thr=0.5):
        super(SmoothL1Loss, self).__init__()
        self.center_R = center_R
        self.weight_rate = weight_rate
        self.score_thr = score_thr

    def forward_score(self, cls_input, loc_input, center_rate):
        cls_part = cls_input.reshape(cls_input.shape[0], -1)
        topk_indices = torch.topk(cls_part, 1, dim=1).indices
        cls_topk_mask = torch.zeros_like(cls_part, dtype=torch.bool)
        for i in range(topk_indices.shape[0]):
            cls_topk_mask[i][topk_indices[i]] = True
        # pos_mask = pos_mask.reshape(-1)
        # pos_mask = cls_topk_mask.reshape(-1) & pos_mask
        pos_mask = cls_topk_mask.reshape(-1)
        bias_map = self.create_loc_bias(cls_input.size(), center_rate, cls_input.device)
        bias_map = bias_map.reshape(-1, 2)
        bias_pos = bias_map[pos_mask].reshape(-1)
        res_pos = loc_input.permute(0, 2, 3, 1).contiguous().reshape(-1, 2)
        res_pos = res_pos[pos_mask].reshape(-1)
        loc_loss = F.smooth_l1_loss(res_pos, bias_pos, reduction='mean')
        return loc_loss * self.weight_rate

    def forward(self, cls_input, loc_input, center_rate):
        if self.center_R == 2:
            target = create_labels_2(cls_input.size(), center_rate, self.center_R)
        else:
            target = create_labels(cls_input.size(), center_rate, self.center_R)
        reg_mask = (target == 1).reshape(-1)
        if self.score_thr <= 0:
            cls_mask = torch.ones_like(reg_mask, dtype=torch.bool)
        else:
            cls_mask = cls_input.sigmoid().reshape(-1) > self.score_thr
        mask = reg_mask & cls_mask
        bias_map = self.create_loc_bias(cls_input.size(), center_rate, cls_input.device)
        bias_map = bias_map.reshape(-1, 2)
        pos_bias_map = bias_map[mask]
        res_pos_map = loc_input.permute(0, 2, 3, 1).contiguous().reshape(-1, 2)
        res_pos_map = res_pos_map[mask].reshape(-1, 2)
        if mask.sum()==0:
            return torch.tensor(0).to(cls_input.device)
        loc_loss = F.smooth_l1_loss(res_pos_map, pos_bias_map, reduction='mean')
        return loc_loss * self.weight_rate

    def create_loc_bias(self, size, center_rate, device):
        b, c, h, w = size
        x = np.arange(h)
        y = np.arange(w)
        xx, yy = np.meshgrid(x, y)
        xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
        loc_map = np.expand_dims(np.concatenate((yy, xx), axis=-1), axis=0).repeat(b, axis=0)
        loc_map_normal = loc_map / [h-1, w-1]

        # calculate the bias and generate the bias map
        center_rate = np.stack(center_rate, axis=1)
        center_rate = center_rate[:, np.newaxis, np.newaxis, :]
        # gt = center_rate.repeat(size[0], axis=1).repeat(size[1], axis=2)
        bias_map = center_rate - loc_map_normal
        bias_map *= [h-1, w-1]
        return torch.from_numpy(bias_map).to(device=device, dtype=torch.float32)


class PeakAwareOffsetLoss(nn.Module):
    def __init__(
        self,
        center_R,
        weight_rate=1.0,
        dense_weight=1.0,
        peak_weight=0.5,
        score_power=1.0,
        peak_score_thr=0.0,
        peak_radius=None,
        topk=1,
        detach_cls_score=True,
    ):
        super(PeakAwareOffsetLoss, self).__init__()
        self.center_R = center_R
        self.weight_rate = weight_rate
        self.dense_weight = dense_weight
        self.peak_weight = peak_weight
        self.score_power = score_power
        self.peak_score_thr = peak_score_thr
        self.peak_radius = peak_radius if peak_radius is not None else center_R
        self.topk = topk
        self.detach_cls_score = detach_cls_score

    def forward(self, cls_input, loc_input, center_rate):
        target = self._create_target(cls_input.size(), center_rate)
        bias_map = self.create_loc_bias(cls_input.size(), center_rate, cls_input.device)
        pred_map = loc_input.permute(0, 2, 3, 1).contiguous()

        cls_score = torch.sigmoid(cls_input)
        score_for_weight = cls_score.detach() if self.detach_cls_score else cls_score

        dense_loss = self._dense_local_loss(score_for_weight, pred_map, bias_map, target)
        peak_loss = self._peak_alignment_loss(score_for_weight, pred_map, bias_map)

        return self.weight_rate * (self.dense_weight * dense_loss + self.peak_weight * peak_loss)

    def _create_target(self, size, center_rate):
        if self.center_R == 2:
            return create_labels_2(size, center_rate, self.center_R)
        return create_labels(size, center_rate, self.center_R)

    def _dense_local_loss(self, score_map, pred_map, bias_map, target):
        pos_mask = (target == 1).squeeze(1)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device)

        bias_norm = torch.norm(bias_map, dim=-1)
        center_weight = torch.exp(-bias_norm / max(float(self.center_R), 1.0))
        conf_weight = score_map.squeeze(1).clamp_min(1e-6).pow(self.score_power)
        weights = pos_mask.float() * center_weight * (0.5 + conf_weight)

        offset_error = F.smooth_l1_loss(pred_map, bias_map, reduction="none").mean(dim=-1)
        return (offset_error * weights).sum() / weights.sum().clamp_min(1e-6)

    def _peak_alignment_loss(self, score_map, pred_map, bias_map):
        b, _, h, w = score_map.shape
        flat_score = score_map.reshape(b, -1)
        flat_pred = pred_map.reshape(b, -1, 2)
        flat_bias = bias_map.reshape(b, -1, 2)

        topk = min(self.topk, flat_score.shape[1])
        topk_score, topk_indices = torch.topk(flat_score, topk, dim=1)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        pred_top = torch.gather(flat_pred, 1, gather_index)
        bias_top = torch.gather(flat_bias, 1, gather_index)

        offset_error = F.smooth_l1_loss(pred_top, bias_top, reduction="none").mean(dim=-1)
        peak_distance = torch.norm(bias_top, dim=-1)
        valid_mask = peak_distance <= float(self.peak_radius)
        if self.peak_score_thr > 0:
            valid_mask = valid_mask & (topk_score >= self.peak_score_thr)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_map.device)

        peak_weights = (0.5 + topk_score.pow(self.score_power)) * valid_mask.float()
        return (offset_error * peak_weights).sum() / peak_weights.sum().clamp_min(1e-6)

    def create_loc_bias(self, size, center_rate, device):
        b, c, h, w = size
        x = np.arange(h)
        y = np.arange(w)
        xx, yy = np.meshgrid(x, y)
        xx, yy = np.expand_dims(xx, -1), np.expand_dims(yy, -1)
        loc_map = np.expand_dims(np.concatenate((yy, xx), axis=-1), axis=0).repeat(b, axis=0)
        loc_map_normal = loc_map / [h - 1, w - 1]

        center_rate = np.stack(center_rate, axis=1)
        center_rate = center_rate[:, np.newaxis, np.newaxis, :]
        bias_map = center_rate - loc_map_normal
        bias_map *= [h - 1, w - 1]
        return torch.from_numpy(bias_map).to(device=device, dtype=torch.float32)


if __name__ == "__main__":
    loss = SmoothL1Loss(center_R=5)
    cls_input = torch.randn((4, 1, 24, 24)).cuda()
    reg_input = torch.randn((4, 2, 24, 24)).cuda()
    center_rate = [[0.5, 0.2, 0.8, 0.2], [0.5, 0.5, 0.5, 0.2]]
    loc_loss = loss(cls_input, reg_input, center_rate)
