# -*- coding: utf-8 -*-
import sys
sys.path.append('/data/gluon-cv/')
"Validate FCOS end to end."
import os
import argparse
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import glob
import logging
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.fcos import FCOSDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from IPython import embed


def parse_args():
    parser = argparse.ArgumentParser(description='Validate FCOS networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base feature extraction network name")
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Validation dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-json', action='store_true',
                        help='Save coco output json')
    parser.add_argument('--eval-all', action='store_true',
                        help='Eval all models begins with save prefix. Use with pretrained.')

    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval',
                                         cleanup=not args.save_json)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(net, val_dataset, val_transform, batch_size, num_workers):
    """Get dataloader."""
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(4)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size, net.base_stride)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return val_loader


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, size):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    nms_thresh = net.nms_thresh
    nms_topk = net.nms_topk
    save_topk = net.save_topk
    with tqdm(total=size) as pbar:
        for batch in val_data:
            batch = split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y, cor, im_scale in zip(*batch):
                # get prediction results
                cls_probs, bboxes = net(x)
                cls_id = cls_probs.argmax(axis=-1)
                probs = mx.nd.pick(cls_probs, cls_id)
                bboxes = net.box_converter(bboxes, cor)
                bboxes = clipper(bboxes.squeeze(axis=0), x)
                im_scale = im_scale.reshape((-1)).asscalar()
                bboxes *= im_scale
                cls_id = cls_id.squeeze(axis=0)
                probs = probs.squeeze(axis=0)
                bboxes = bboxes.squeeze(axis=0)
                target = mx.nd.concat(cls_id.expand_dims(axis=1),
                            probs.expand_dims(axis=1), bboxes, dim=-1)
                keep = mx.nd.contrib.box_nms(target, overlap_thresh=nms_thresh, coord_start=2,
                                             topk=nms_topk, valid_thresh=0.00001, score_index=1,
                                             id_index=0, force_suppress=False,
                                             in_format='corner', out_format='corner')
                keep = keep[:save_topk].expand_dims(axis=0)
                det_ids.append(keep.slice_axis(axis=-1, begin=0, end=1))
                det_scores.append(keep.slice_axis(axis=-1, begin=1, end=2))
                det_bboxes.append(keep.slice_axis(axis=-1, begin=2, end=None))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_bboxes[-1] *= im_scale
                gt_difficults.append(
                        y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            # update metric
            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in \
                zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
                eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
            pbar.update(len(ctx))
    return eval_metric.get()


if __name__ == '__main__':
    args = parse_args()

    # validation contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx) # 1 batch per device

    # network
    kwargs = {}
    net_name = "_".join(("fcos", args.network, args.dataset))
    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, pretrained=True, **kwargs)
    else:
        net = gcv.model_zoo.get_model(net_name, pretrained=False, **kwargs)
        net.load_parameters(args.pretrained.strip())
    net.collect_params().reset_ctx(ctx)

    # validation data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        net, val_dataset, FCOSDefaultValTransform, args.batch_size, args.num_workers)

    # validation
    if not args.eval_all:
        names, values = validate(net, val_data, ctx, eval_metric, len(val_dataset))
        for k, v in zip(names, values):
            print(k, v)
    else:
        saved_models = glob.glob(args.save_prefix + '*.params')
        for epoch. saved_model in enumerate(sorted(saved_models)):
            print('[Epoch {}] Validating from {}'.format(epoch, saved_model))
            net.load_parameters(saved_model)
            net.collect_params().reset_ctx(ctx)
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, len(val_dataset))
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            with open(args.save_prefix+'_best_map.log', 'a') as f:
                f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
