_base_ = '../yolox/yolox_s_8xb8-300e_coco.py'

max_epochs = 300
num_last_epochs = 5

train_cfg = dict(max_epochs=max_epochs)

# model settings
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            random_size_range=(320, 640),
            size_divisor=32,
            interval=10)
    ]),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(num_classes=1, in_channels=96, feat_channels=96))

img_scale = (640, 640)  # width, height

dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('fish', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # Resize and Pad are for the last 15 epochs when Mosaic and
    # RandomAffine are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(416, 416), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root='/home/cycyang/NIWA/mmdetection/data/',
        ann_file='/home/cycyang/NIWA/mmdetection/data/20221207_coco_format_annotation_train.json',
        data_prefix=dict(img='20221207_train/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
     	),
    pipeline=train_pipeline)


train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root='/home/cycyang/NIWA/mmdetection/data/',
        ann_file='/home/cycyang/NIWA/mmdetection/data/20230614_coco_format_annotation_val.json',
        data_prefix=dict(img='20230614_val/'),
        test_mode=True,
        pipeline=test_pipeline,
        ))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file='/home/cycyang/NIWA/mmdetection/data/20230614_coco_format_annotation_val.json')
test_evaluator = val_evaluator

load_from = "checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"

