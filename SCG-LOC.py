# 1. 模型配置(models) =========================================
model = dict(
    backbone=dict(
        type='MixFormer',
        vit_type="cvt13",
        pretrain_path="pretrained/CvT-13-384x384-IN-22k.pth",
        pretrain=True,
        output_index=[0, 1, 2]
    ),
    neck=dict(
        type="DPFPN",
        output_dims=128,
        rfa_hidden_ratio=4,
        fusion_hidden_ratio=4,
        detail_hidden_ratio=4,
        gate_bias=2.0,
        UAV_output_index=[0],
        Satellite_ouput_index=0,
    ),
    head=dict(
        type="ChannelEmbedding",
        input_ndim=128,
        mid_process_channels=[64, 16, 1]
    ),
    dspm=dict(
        enable=True,
        channels=128,
        match_dim=96,         ######### 64
        token_grid=3,
        hidden_channels=128,
        norm_groups=32,
        temperature=0.01,      ####### 0.1
        dual_softmax=True,
        gate_bias=-1.0,
        alpha_init=0.0,
    ),
    postprocess=dict(
        upsample_to_original=True,
        upsample_method="NearstUpsample",
        output_size=[384, 384],
    ),
    loss=dict(
        cls_loss=dict(
            type="BalanceLoss",
            center_R=33,
            neg_weight=170
        ),
    )
)


# 2. 数据集配置(datasets) =========================================
data_config = dict(
    batchsize=8,
    num_worker=8,
    val_batchsize=8,
    train_dir="data/map2019/train",
    val_dir="data/map2019/merge_test_700-1800_cr0.95_stride100",
    test_dir="data/map2019",
    test_mode="merge_test_700-1800_cr0.95_stride100",
    UAVhw=[128, 128],
    Satellitehw=[384, 384],
)
pipline_config = dict(
    train_pipeline=dict(
        UAV=dict(
            RandomErasing=dict(probability=0.2),
            RandomResize=dict(img_size=data_config['UAVhw']),
            ToTensor=dict(),
        ),
        Satellite=dict(
            RandomCrop=dict(cover_rate=0.90, map_size=(512, 1500)),
            RandomResize=dict(img_size=data_config['Satellitehw']),
            ToTensor=dict(),
        ),
    ),
)

# 3. 训练策略配置(schedules) =========================================
lr_config = dict(
    lr=1.2e-4,
    type='cosine',
    warmup_iters=800,
    warmup_ratio=0.01,
)
train_config = dict(
    autocast=True,
    num_epochs=90
)
test_config = dict(
    num_worker=8,
    filterR=33,
    checkpoint="output/net_best.pth",
)

# 4. 运行配置(runtime) =========================================
checkpoint_config = dict(
    interval=2,
    epoch_start_save=50,
    only_save_best=True,
)
log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666
