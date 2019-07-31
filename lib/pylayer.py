#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code written by KAI ZHAO (http://kaiz.xyz)
import caffe
import numpy as np
from os.path import join, isfile, splitext
import random, cv2
import augim

class ImageLabelmapDataLayer(caffe.Layer):
  """
  Python data layer
  """
  def setup(self, bottom, top):
    params = eval(self.param_str)
    self.root = params['root']
    self.source = params['source']
    self.shuffle = bool(params['shuffle'])
    self.mean = np.array((104.00699, 116.66877, 122.67892))
    self.aug = False
    if 'aug' in params:
      self.aug = bool(params['aug'])
    with open(join(self.root, self.source), 'r') as f:
      self.filelist = f.readlines()
    if self.shuffle:
      random.shuffle(self.filelist)
    self.idx = 0
    top[0].reshape(1, 3, 100, 100) # im
    top[1].reshape(1, 1, 100, 100) # lb
  
  def reshape(self, bottom, top):
    """
    Will reshape in forward()
    """

  def forward(self, bottom, top):
    """
    Load data
    """
    filename = splitext(self.filelist[self.idx])[0]
    imfn = join(self.root, 'images', filename+".jpg")
    lbfn = join(self.root, 'annotations', filename+".png")
    assert isfile(imfn), "file %s doesn't exist!" % imfn
    assert isfile(lbfn), "file %s doesn't exist!" % lbfn
    im = cv2.imread(imfn).astype(np.float32)
    lb = cv2.imread(lbfn, 0).astype(np.float32)
    if self.aug:
      im, lb = augim.rescale([im, lb], np.linspace(0.5, 1.5, 11))
      if np.random.binomial(1, 0.2):
        im, lb = augim.rotate([im, lb], angle=[-10, 10], expand=False)
      im, lb = augim.fliplr([im, lb])
    assert np.unique(lb).size == 2, "unique(lb).size = %d" % np.unique(lb).size
    lb[lb != 0] = 1
    im, lb = map(lambda x:np.float32(x), [im, lb])
    if im.ndim == 2:
      im = im[:,:,np.newaxis]
      im = np.repeat(im, 3, 2)
    im -= self.mean
    im = np.transpose(im, (2, 0, 1))
    im = im[np.newaxis, :, :, :]
    assert lb.ndim == 2, "lb.ndim = %d" % lb.ndim
    h, w = lb.shape
    assert im.shape[2] == h and im.shape[3] == w, "Image and GT shape mismatch."
    lb = lb[np.newaxis, np.newaxis, :, :]
    if np.count_nonzero(lb) == 0:
      print "Warning: all zero label map!"
    top[0].reshape(1, 3, h, w)
    top[1].reshape(1, 1, h, w)
    top[0].data[...] = im
    top[1].data[...] = lb
    if self.idx == len(self.filelist)-1:
      # we've reached the end, restart.
      print "Restarting data prefetching from start."
      if self.shuffle:
        random.shuffle(self.filelist)
      self.idx = 0
    else:
      self.idx = self.idx + 1

  def backward(self, top, propagate_down, bottom):
    """
    Data layer doesn't need back propagate
    """
    pass
