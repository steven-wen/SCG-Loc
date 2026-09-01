import torch.nn as nn
import numpy as np
from .Backbone.backbone import make_backbone
from .Neck.neck import make_neck
from .Neck.dspm import DSPM
from .Head.head import make_head
from .PostProcess.postprocess import make_postprocess
import time
import torch
from torch.nn import functional as F


class FPI(nn.Module):
    def __init__(self, opt):
        super(FPI, self).__init__()
        self.opt = opt
        # backbone init
        self.backbone_name = opt.model["backbone"]["type"]
        if self.backbone_name == "MixFormer":
            opt.model["backbone"]["satellite_size"] = opt.data_config["Satellitehw"][0]
            opt.model["backbone"]["uav_size"] = opt.data_config["UAVhw"][0]
            self.union_backbone = make_backbone(opt)
            opt.backbone_output_channel = self.union_backbone.backbone_out_channel
        else:
            self.backbone_uav = make_backbone(opt, opt.data_config["UAVhw"])
            self.backbone_satellite = make_backbone(opt, opt.data_config["Satellitehw"])
            opt.backbone_output_channel = self.backbone_uav.backbone_out_channel

        # neck init
        self.neck_uav = make_neck(opt)
        self.neck_satellite = make_neck(opt)
        dspm_opt = opt.model.get("dspm", {})
        self.dspm = None
        if bool(dspm_opt.get("enable", False)):
            self.dspm = DSPM(
                channels=dspm_opt.get("channels", opt.model["neck"].get("output_dims", 128)),
                match_dim=dspm_opt.get("match_dim", 64),
                token_grid=dspm_opt.get("token_grid", 3),
                hidden_channels=dspm_opt.get("hidden_channels", opt.model["neck"].get("output_dims", 128)),
                norm_groups=dspm_opt.get("norm_groups", 32),
                temperature=dspm_opt.get("temperature", 0.1),
                dual_softmax=dspm_opt.get("dual_softmax", True),
                gate_bias=dspm_opt.get("gate_bias", -1.0),
                alpha_init=dspm_opt.get("alpha_init", 0.0),
                center_prior=dspm_opt.get("center_prior", False),
                center_prior_min=dspm_opt.get("center_prior_min", 0.5),
                center_prior_power=dspm_opt.get("center_prior_power", 2.0),
                center_prior_on=dspm_opt.get("center_prior_on", "value"),
                global_query_score=dspm_opt.get("global_query_score", False),
                global_query_score_weight=dspm_opt.get("global_query_score_weight", 0.5),
                global_query_pool_alpha=dspm_opt.get("global_query_pool_alpha", 0.1),
                p3_update_scale=dspm_opt.get("p3_update_scale", 1.0),
                update_p4=dspm_opt.get("update_p4", False),
                p4_update_scale=dspm_opt.get("p4_update_scale", 0.5),
            )
        self.UAV_output_index = opt.model["neck"]["UAV_output_index"]
        self.Satellite_output_index = opt.model["neck"]["Satellite_ouput_index"]

        # head init
        opt.model["head"]["muti_level_nums"] = len(opt.model["neck"]["UAV_output_index"])
        self.head = make_head(opt)

        # upsample init
        self.upsample_to_original = opt.model["postprocess"]["upsample_to_original"]
        if self.upsample_to_original:
            self.postprocess = make_postprocess(opt)

    def forward(self, z, x):
        # backbone forward
        if self.backbone_name == "MixFormer":
            z, x = self.union_backbone(x, z)
        else:
            z = self.backbone_uav(z)
            x = self.backbone_satellite(x)
        # neck forward
        neck_z_multi = self.neck_uav(z)
        neck_z = [neck_z_multi[index] for index in self.UAV_output_index]
        guidance_features = z if getattr(self.neck_satellite, "requires_uav_guidance", False) else None
        neck_x_multi = self.neck_satellite(x, guidance_features=guidance_features)
        if self.dspm is not None:
            neck_x_multi = self.dspm(neck_x_multi, neck_z_multi)
        neck_x = neck_x_multi[self.Satellite_output_index]
        # head forward
        cls_out, reg_out = self.head(neck_z, neck_x)
        # postprocess forward
        if self.upsample_to_original:
            cls_out,reg_out = self.postprocess(cls_out,reg_out)

        return cls_out, reg_out

    def load_checkpoint(self, checkpoint_path=""):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)


def make_model(opt):
    # init the FPI model
    model = FPI(opt)
    # if 'load_from' is not empty, load the pretrain checkpoint.
    if isinstance(opt.load_from, str) and len(opt.load_from) > 0:
        model.load_checkpoint(opt.load_from)
    return model
