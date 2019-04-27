"""Transforms for RCNN series."""
from __future__ import absolute_import

import copy
from random import randint

import numpy as np
import mxnet as mx
from mxnet import nd

from .. import bbox as tbbox
from .. import image as timage
from .. import mask as tmask

__all__ = ['transform_test', 'load_test',
           'FCOSDefaultTrainTransform', 'FCOSDefaultValTransform']

def transform_test(imgs, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [mx.image.imread(f) for f in filenames]
    return transform_test(imgs, short, max_size, mean, std)


class FCOSDefaultTrainTransform(object):
    def __init__(self, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), flip_p=0.5, retina_stages=5,
                 base_stride=8, num_class=80, **kwargs):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._random_resize = isinstance(self._short, (tuple, list))
        self._flip_p = flip_p
        self._num_class = num_class

        from ....model_zoo.fcos.fcos_target import FCOSTargetGenerator
        self._target_generator = FCOSTargetGenerator(retina_stages=retina_stages,
                                    base_stride=base_stride)

    def __call__(self, src, label):
        "Apply transform to training image/label."
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        if self._random_resize:
            short = randint(self._short(0), self._short[1])
        else:
            short = self._short
        img = timage.resize_short_within(src, short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=self._flip_p)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # generate training targets for fcos
        tbox = mx.nd.array(bbox)
        cls_targets, ctr_targets, box_targets, cor_targets = self._target_generator(img, tbox)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        return img, cls_targets, ctr_targets, box_targets, cor_targets


class FCOSDefaultValTransform(object):
    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))
        im_scale = h / float(img.shape[0])

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32'), mx.nd.array([im_scale])
