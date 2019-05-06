# -*- coding: utf-8 -*-
"""Fully Convolutional One-Stage Object Detection."""
from __future__ import absolute_import

import os
import warnings

import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from IPython import embed

from .fcos_target import FCOSBoxConverter
from ...nn.feature import RetinaFeatureExpander

__all__ = ['FCOS', 'get_fcos',
           'fcos_resnet50_v1_coco',
           'fcos_resnet50_v1b_coco',
           'fcos_resnet101_v1d_coco']


class ConvPredictor(nn.HybridBlock):
    def __init__(self, num_channels, share_params=None, bias_init=None, **kwargs):
        super(ConvPredictor, self).__init__(**kwargs)
        with self.name_scope():
            if share_params is not None:
                self.conv = nn.Conv2D(num_channels, 3, 1, 1, params=share_params,
                        bias_initializer=bias_init)
            else:
                self.conv = nn.Conv2D(num_channels, 3, 1, 1,
                        weight_initializer=mx.init.Normal(sigma=0.01),
                        bias_initializer=bias_init)

    def get_params(self):
        return self.conv.params

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


class RetinaHead(nn.HybridBlock):
    def __init__(self, share_params=None, **kwargs):
        super(RetinaHead, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            for i in range(4):
                if share_params is not None:
                    self.conv.add(nn.Conv2D(256, 3, 1, 1, activation='relu',
                        params=share_params[i]))
                else:
                    self.conv.add(nn.Conv2D(256, 3, 1, 1, activation='relu',
                        weight_initializer=mx.init.Normal(sigma=0.01),
                        bias_initializer='zeros'))

    def _share_params(self, conv1, conv2):
        """share conv2 weights and grad to conv1"""
        conv1 = nn.Conv2D(256, 3, 1, 1, params=conv2.params)
        return conv1

    def set_params(self, newconv):
        for b, nb in zip(self.conv, newconv):
            b.weight.set_data(nb.weight.data())
            b.bias.set_data(nb.bias.data())

    def get_params(self):
        param_list = []
        for opr in self.conv:
            param_list.append(opr.params)
        return param_list

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


@mx.init.register
class ClsBiasInit(mx.init.Initializer):
    def __init__(self, num_class, cls_method="sigmoid", pi=0.01, **kwargs):
        super(ClsBiasInit, self).__init__(**kwargs)
        self._num_class = num_class
        self._cls_method = cls_method
        self._pi = pi

    def _init_weight(self, name, data):
        if self._cls_method == "sigmoid":
            arr = -1 * np.ones((data.size, ))
            arr = arr *  np.log((1 - self._pi) / self._pi)
            data[:] = arr
        elif self._cls_method == "softmax":
            pass


class FCOS(nn.HybridBlock):
    """Fully Convolutional One-Stage Object Detection."""
    def __init__(self, features, classes, base_stride=128, short=600, max_size=1000,
                 valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                 retina_stages=5, pretrained=False, norm_layer=None, norm_kwargs=None,
                 nms_thresh=0.5, nms_topk=1000, save_topk=100, ctx=mx.cpu(), **kwargs):
        super(FCOS, self).__init__(**kwargs)
        self.short = short
        self.max_size = max_size
        self.base_stride = base_stride
        self.valid_range = valid_range
        self.classes = len(classes)
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.save_topk = save_topk
        self._retina_stages = retina_stages

        with self.name_scope():
            bias_init = ClsBiasInit(self.classes)
            cls_heads = nn.HybridSequential()
            box_heads = nn.HybridSequential()
            cls_preds = nn.HybridSequential()
            ctr_preds = nn.HybridSequential()
            box_preds = nn.HybridSequential()
            share_cls_params, share_box_params = None, None
            share_cls_pred_params, share_ctr_pred_params, \
                        share_box_pred_params = None, None, None
            for i in range(self._retina_stages):
                # cls
                cls_head = RetinaHead(share_params=share_cls_params)
                cls_heads.add(cls_head)
                share_cls_params = cls_head.get_params()
                # box
                box_head = RetinaHead(share_params=share_box_params)
                box_heads.add(box_head)
                share_box_params = box_head.get_params()
                # cls preds
                cls_pred = ConvPredictor(num_channels=self.classes,
                                share_params=share_cls_pred_params, bias_init=bias_init)
                cls_preds.add(cls_pred)
                share_cls_pred_params = cls_pred.get_params()
                # ctr preds
                ctr_pred = ConvPredictor(num_channels=1,
                                share_params=share_ctr_pred_params, bias_init='zeros')
                ctr_preds.add(ctr_pred)
                share_ctr_pred_params = ctr_pred.get_params()
                # box preds
                box_pred = ConvPredictor(num_channels=4,
                                share_params=share_box_pred_params, bias_init='zeros')
                box_preds.add(box_pred)
                share_box_pred_params = box_pred.get_params()

            self._cls_heads = cls_heads
            self._box_heads = box_heads
            self._cls_preds = cls_preds
            self._ctr_preds = ctr_preds
            self._box_preds = box_preds

            # self.scale_list = [self.params.get('scale_p{}'.format(i), shape=(1,),
            #                                   differentiable=True, allow_deferred_init=True,
            #                                   init='ones') for i in range(retina_stages)]

            self._retina_features = features
            self.box_converter = FCOSBoxConverter()

    def _get_fcos_boxes(self, box_pred):
        """return box preds to corner format,
           this is for IoU loss computation.

           box_pred : [B, 4, H, W]
        """
        pass

    def hybrid_forward(self, F, x):
        """make fcos heads
        x : [B, C, H, W]

        Return:
        -------
        cls_preds : [B, N, C]
        ctr_preds : [B, N ,1]
        box_preds : [B, N, 4]
        """
        cls_preds_list = []
        ctr_preds_list = []
        box_preds_list = []
        stride = self.base_stride
        retina_fms = self._retina_features(x)
        for i in range(self._retina_stages):
            x = retina_fms[i]
            fm_cls = x
            fm_cls = self._cls_heads[i](fm_cls)
            cls_pred = self._cls_preds[i](fm_cls)
            ctr_pred = self._ctr_preds[i](fm_cls)
            # cls_pred = F.sigmoid(cls_pred)
            cls_pred = F.reshape(F.transpose(cls_pred, (0, 2, 3, 1)), (0, -1, self.classes))
            ctr_pred = F.reshape(F.transpose(ctr_pred, (0, 2, 3, 1)), (0, -1, 1))

            fm_box = x
            fm_box = self._box_heads[i](fm_box)
            box_pred = self._box_preds[i](fm_box)
            # TODO@ANG: fix bugs in scale param
            # box_pred = F.exp(F.broadcast_mul(scale_list[i], box_pred) * stride
            box_pred = F.exp(box_pred) * stride
            box_pred = F.reshape(F.transpose(box_pred, (0, 2, 3, 1)), (0, -1, 4))

            cls_preds_list.append(cls_pred)
            ctr_preds_list.append(ctr_pred)
            box_preds_list.append(box_pred)
            stride /= 2

        cls_preds = F.concat(*cls_preds_list, dim=1)
        ctr_preds = F.concat(*ctr_preds_list, dim=1)
        box_preds = F.concat(*box_preds_list, dim=1)

        if autograd.is_training():
            return cls_preds, ctr_preds, box_preds
        else:
            # [B, N, C], [B, N, 1], [B, N, 4]
            cls_prob = F.sigmoid(cls_preds)
            ctr_prob = F.sigmoid(ctr_preds)
            cls_prob = F.broadcast_mul(cls_prob, ctr_prob)
            # [N, C], [N, 4]
            return cls_prob, box_preds


def get_fcos(name, dataset, pretrained=False, ctx=mx.cpu(),
             root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    "return FCOS network"
    net = FCOS(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('fcos', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net


def fcos_resnet50_v1_coco(pretrained=False, pretrained_base=True, **kwargs):
    from ..resnet import resnet50_v1
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1(pretrained=pretrained_base, **kwargs)
    features = RetinaFeatureExpander(network=base_network,
                                     pretrained=pretrained_base,
                                     outputs=['stage2_activation3',
                                              'stage3_activation5',
                                              'stage4_activation2'])
    # out = features(mx.sym.var('data'))
    # for o in out:
    #     print(o.infer_shape(data=(1, 3, 562, 1000))[1][0])
    return get_fcos(name="resnet50_v1", dataset="coco", pretrained=pretrained,
                    features=features, classes=classes, base_stride=128, short=800,
                    max_size=1333, norm_layer=None, norm_kwargs=None,
                    valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                    nms_thresh=0.5, nms_topk=1000, save_topk=100)


def fcos_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = RetinaFeatureExpander(network=base_network,
                                     pretrained=pretrained_base,
                                     outputs=['layers2_relu11_fwd',
                                              'layers3_relu17_fwd',
                                              'layers4_relu8_fwd'])
    # out = features(mx.sym.var('data'))
    # for o in out:
    #     print(o.infer_shape(data=(1, 3, 562, 1000))[1][0])
    return get_fcos(name="resnet50_v1b", dataset="coco", pretrained=pretrained,
                    features=features, classes=classes, base_stride=128, short=800,
                    max_size=1333, norm_layer=None, norm_kwargs=None,
                    valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                    nms_thresh=0.5, nms_topk=1000, save_topk=100)


def fcos_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    from ..resnetv1b import resnet101_v1d
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = RetinaFeatureExpander(network=base_network,
                                     pretrained=pretrained_base,
                                     outputs=['layers2_relu11_fwd',
                                              'layers3_relu68_fwd',
                                              'layers4_relu8_fwd'])
    return get_fcos(name="resnet101_v1d", dataset="coco", pretrained=pretrained,
                    features=features, classes=classes, base_stride=128, short=800,
                    max_size=1333, norm_layer=None, norm_kwargs=None,
                    valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                    nms_thresh=0.5, nms_topk=1000, save_topk=100)


if __name__ == '__main__':
    net = fcos_resnet50_v1b_coco(pretrained_base=True, ctx=mx.gpu(0))
    from IPython import embed; embed()
