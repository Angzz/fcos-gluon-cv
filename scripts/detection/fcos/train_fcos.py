# -*- coding: utf-8 -*-
import sys
sys.path.append('/data/gluon-cv/')
"Train FCOS end to end."
import os
import argparse

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.fcos import \
        FCOSDefaultTrainTransform, FCOSDefaultValTransform
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description='Train FCOS networks e2e.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, '
                                        'if your CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    parser.add_argument('--mixup', action='store_true', help='Use mixup training.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')

    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')

    args = parser.parse_args()
    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '12,16'
        args.lr = float(args.lr) if args.lr else 0.00125
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 8000
        args.wd = float(args.wd) if args.wd else 1e-4
        num_gpus = len(args.gpus.split(','))
        if num_gpus == 1:
            args.lr_warmup = -1
        else:
            args.lr *= num_gpus
            args.lr_warmup /= num_gpus
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_workers):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(train_transform(
            net.short, net.max_size, net.base_stride, net.valid_range)),
            batch_size, True, batchify_fn=train_bfn, last_batch='rollover',
            num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(4)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size, net.base_stride)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def get_lr_at_iter(alpha):
    return 1. / 10. * (1 - alpha) + alpha


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    nms_thresh = net.nms_thresh
    nms_topk = net.nms_topk
    save_topk = net.save_topk
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
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    # net.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(
        net.collect_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': args.momentum})

    # lr_decay_policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)

    # losses and metrics
    fcos_cls_loss = gcv.loss.SigmoidFocalLoss(
            from_logits=False, sparse_label=True, num_class=net.classes+1)
    fcos_ctr_loss = gcv.loss.CtrNessLoss()
    fcos_box_loss = gcv.loss.IOULoss(return_iou=False)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        mix_ratio = 1.0
        if args.mixup:
            train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset._data.set_mixup(None)
                mix_ratio = 1.0
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        tic = time.time()
        btic = time.time()
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            cls_losses = []
            ctr_losses = []
            box_losses = []
            with autograd.record():
                # per card
                for data, cls_target, ctr_target, box_target, cor_target in zip(*batch):
                    if cls_target.sum().asscalar() == 0:
                        batch_size -= int(data.shape[0])
                        # TODO@ANG: fix the lr if data is invalid
                        # trainer.set_learning_rate()
                        continue
                    # [B, N, C], [B, N, 1], [B, N, 4]
                    cls_pred, ctr_pred, box_pred = net(data)
                    box_pred = net.box_converter(box_pred, cor_target)
                    cls_loss = fcos_cls_loss(cls_pred, cls_target)
                    ctr_pred = ctr_pred.squeeze(axis=-1)
                    ctr_loss = fcos_ctr_loss(ctr_pred, ctr_target, cls_target)
                    box_loss = fcos_box_loss(box_pred, box_target, cls_target)
                    loss = cls_loss + ctr_loss + box_loss
                    losses.append(loss)
                    cls_losses.append(cls_loss.asscalar())
                    ctr_losses.append(ctr_loss.asscalar())
                    box_losses.append(box_loss.asscalar())
                autograd.backward(losses)
            trainer.step(batch_size) # normalize by batch_size
            if args.log_interval and not (i + 1) % args.log_interval:
                total_cls_loss = np.array(cls_losses, dtype=np.float32).mean()
                total_ctr_loss = np.array(ctr_losses, dtype=np.float32).mean()
                total_box_loss = np.array(box_losses, dtype=np.float32).mean()
                print_loss = {'cls_loss': total_cls_loss, 'ctr_loss': total_ctr_loss, \
                        'box_loss': total_box_loss}
                msg = ', '.join(['{}={:.3f}'.format(k, v) for k, v in print_loss.items()])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'\
                        .format(epoch, i, args.log_interval * batch_size / (time.time() \
                        - btic), msg))
                btic = time.time()
        logger.info('[Epoch {}] Training cost: {:.3f}'.format(
            epoch, (time.time() - tic)))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(1100)

    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    kwargs = {}
    net_name = "_".join(("fcos", args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True, **kwargs)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            if 'scale' in param.name:
                param._data = mx.nd.ones(1)
            else:
                param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
            net, train_dataset, val_dataset, FCOSDefaultTrainTransform,
            FCOSDefaultValTransform, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
