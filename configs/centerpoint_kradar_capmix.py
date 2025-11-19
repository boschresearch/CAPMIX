import copy
import os
_base_ = [
    '../../../configs/_base_/default_runtime.py',
]


# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0.,-16.,-2.,80.,16.,7.6]
post_center_range_m = [sum(x) for x in zip(point_cloud_range, [-5., -5., -1., 5., 5., 1.])]
voxel_size = [0.2, 0.2, 0.4]
grid_size_xyz = [int((abs(point_cloud_range[0]) + abs(point_cloud_range[3]))/ voxel_size[0]), 
                  int((abs(point_cloud_range[1]) + abs(point_cloud_range[4]))/ voxel_size[1]), 
                  int((abs(point_cloud_range[2]) + abs(point_cloud_range[5]))/ voxel_size[2])]
grid_shape_zyx = [grid_size_xyz[2] + 1, grid_size_xyz[1], grid_size_xyz[0]]


class_names = ['Sedan','Bus or Truck']
data_root = 'XXX/k-radar/full/'
metainfo = dict(classes=class_names)
num_classes = len(class_names)

input_dim = 4

cpu_cores=num_workers = min(os.cpu_count(), 4)


train_ratio=1
val_ratio=1
test_ratio=1

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            voxel_size=voxel_size,
            max_voxels=(270000, 360000),
            point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=grid_shape_zyx,
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 64), (64, 64, 128), (128,128)),
        encoder_paddings=((0, 0, 1),(0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=["Sedan"]),
            dict(num_class=1, class_names=["Bus or Truck"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=post_center_range_m,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size_xyz,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=1500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=post_center_range_m,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))



traindataset_type = 'KradarDataset'
valdataset_type = 'KradarDataset'
file_client_args = dict(backend='disk')
custom_imports = dict(imports=[
                               'capmix_public.dataset.Kradar',
                                'capmix_public.dataset.loading',
                                'capmix_public.evaluation.KradarMetric',
                                ], allow_failed_imports=False)


train_pipeline = [
    dict(
            type='LoadKradarFrame',roi=point_cloud_range,input_dim=input_dim,),
    dict(type='CAPMix', points_dim=input_dim, prob=0.5, roi=point_cloud_range,
        pre_transform=[
            dict(
                type='LoadKradarFrame',roi=point_cloud_range,input_dim=input_dim,),]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.1, 0.1], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointShuffle"),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['gt_bboxes_3d', 'gt_labels_3d', 'points'])
]
test_pipeline = [
    dict(
        type='LoadKradarFrame',roi=point_cloud_range,input_dim=input_dim,),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadKradarFrame',roi=point_cloud_range,input_dim=input_dim,),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]
folder_name='radar_pointcloud1p1'
train_dataloader=dict(
    batch_size=16,
    num_workers=cpu_cores,
    persistent_workers=False,
    drop_last=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=traindataset_type,
        data_root=data_root,
        ratio=train_ratio,
        with_velocity=False,
        folder_name=folder_name,
        ann_file='XXX/k-radar/full/filtered_traindata.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        eval_mode='kitti',
        test_mode=False,
        point_cloud_range=point_cloud_range,
        box_type_3d='LiDAR'),)

val_dataloader=dict(
    batch_size=16,
    num_workers=cpu_cores,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=valdataset_type,
        pipeline=test_pipeline, 
        classes=class_names, 
        with_velocity=False,
        ratio=test_ratio,
        eval_mode='kitti',
        folder_name=folder_name,
        data_root=data_root,
        point_cloud_range=point_cloud_range,
        ann_file='XXX/k-radar/full/filtered_testdata.pkl',
        metainfo=metainfo,
        test_mode=True,
        ))
test_dataloader=val_dataloader

# lr=1e-4
lr=1e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=20,
        end=28,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=28,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=20,
        end=28,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=28,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]


total_epochs = 40
eval_interval = 1

auto_scale_lr = dict(enable=True, base_batch_size=16)

# --------------kitti evaluation and logger-----------
val_evaluator = dict(
    type='KittiMetricforKradar',
    pcd_limit_range=point_cloud_range,
    class_names=class_names,
    ratio=val_ratio,
    ann_file='XXX/k-radar/full/filtered_testdata.pkl',
    metric='bbox',
    backend_args=None)
test_evaluator = val_evaluator
default_hooks=dict(
    logger=dict(type='LoggerHook', interval=200),
    checkpoint=dict(
        type='CheckpointHook',
        save_best='Kitti metric/pred_instances_3d/BEV_mAP40_0.3',
        rule='greater',
        )
)

visualizer = dict(
    type='Det3DLocalVisualizer',
    name='visualizer')

train_cfg = dict(by_epoch=True, max_epochs=total_epochs, val_interval=eval_interval)
val_cfg = dict()
test_cfg = dict()
# randomness = dict(seed=2024)
randomness = dict(seed=2025)
# randomness = dict(seed=2026)
