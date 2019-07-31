# =================================================
# * Licensed under The MIT License
# * Written by KAI-ZHAO
# =================================================
# AugIm: Image Augmentation
# Usage:
# im = np.zeros((100, 100, 3), dtype=np.uint8)
# label_map = np.zeros((100, 100), dtype=np.uint8)
# im, label_map = augim.rescale([im, label_map], scales=[0.6. 0.8. 1.0. 1.2])
import numpy as np
from PIL import Image
import skimage.transform as transform
import cv2
USE_CV2_RESIZE = True

def resizeim(x, h, w):
  if USE_CV2_RESIZE:
    return cv2.resize(x, (w, h))
  else:
    pil_im = Image.fromarray(x.astype(np.uint8))
    pil_im = pil_im.resize((w, h))
    return np.array(pil_im)

def resizegt(x, h, w):
  """
  resize a BINARY map
  """
  assert np.unique(x).size == 2
  if USE_CV2_RESIZE:
    im1 = cv2.resize(x, (w, h), cv2.INTER_NEAREST)
    thres = float(im1.max() - im1.min()) / 2
    im1[im1 < thres] = 0
    im1[im1 != 0] = 1
    return im1
  else:
    pil_im = Image.fromarray(x)
    pil_im = pil_im.resize((w, h), resample=Image.NEAREST)
    return np.array(pil_im)

def shape_match(im, gt):
  return im.shape[0] == gt.shape[0] and im.shape[1] == gt.shape[1]

#=======================================================# 
# Operations that can be performed on BOTH images and
# label-maps
#=======================================================# 
def rescale(x, scales, keep=True):
  """
  rescale (resize) image and ground-truth map
  x: input image (or image/label pair)
  keep: keep width-heigh ratio or not
  """
  assert isinstance(x, list) or isinstance(x, np.ndarray)
  assert isinstance(scales, list) or isinstance(scales, np.ndarray)
  if isinstance(scales, list):
    scales = np.array(scales)
  s = np.random.choice(scales)
  s1 = np.random.choice(scales)
  if isinstance(x, list):
    assert len(x) == 2
    im, gt = x
    h, w, c = im.shape
    assert c == 3
    assert h == gt.shape[0] and w == gt.shape[1]
    if keep:
      h1, w1 = int(h * s), int(w * s)
    else:
      h1, w1 = int(h * s), int(w * s1)
    return [resizeim(im, h1, w1), resizegt(gt, h1, w1)]
  elif isinstance(x, np.ndarray):
    h, w, c = x.shape
    assert c == 3
    if keep:
      h1, w1 = int(h * s), int(w * s)
    else:
      h1, w1 = int(h * s), int(w * s1)
    return resizeim(x, h, w)
  else:
    raise TypeError("Error!")

def fliplr(x, p=0.5):
  """
  Flip left-right
  """
  assert isinstance(x, list) or isinstance(x, np.ndarray)
  flag = np.random.binomial(1, p)
  if flag:
    if isinstance(x, list):
      im, gt = x
      assert im.ndim == 3 and gt.ndim == 2
      assert im.shape[0] == gt.shape[0] and im.shape[1] == gt.shape[1] \
             and im.shape[2] == 3
      return [im[:, ::-1, :], gt[:, ::-1]]
    else:
      assert x.ndim == 3
      return x[:, ::-1, :]
  else:
    return x

def crop(x, offset=20):
  assert isinstance(x, list) or isinstance(x, np.ndarray)
  assert offset > 0
  if isinstance(x, list):
    im, gt = x
    has_gt = True
  else:
    im = x
    has_gt = False
  h, w, c = im.shape
  if has_gt:
    assert gt.shape[0] == h and gt.shape[1] == w
  assert offset < h // 2 and offset < w // 2
  xstart = np.random.choice(np.arange(1, offset))
  xend = w - np.random.choice(np.arange(1, offset))
  ystart = np.random.choice(np.arange(1, offset))
  yend = h - np.random.choice(np.arange(1, offset))
  if has_gt:
    return [im[ystart:yend, xstart:xend, :], gt[ystart:yend, xstart:xend]]
  else:
    return im[ystart:yend, xstart:xend]

def rescale_crop(x, size=[256 ,256]):
  h0, w0 = map(lambda x: int(x), list(size))
  assert isinstance(x, list) or isinstance(x, np.ndarray)
  r = -1
  if isinstance(x, list):
    im, lb = x
    assert shape_match(im, lb)
    pil_im = Image.fromarray(im.astype(np.uint8))
    pil_lb = Image.fromarray(lb.astype(np.uint8))
  else:
    im = x
    assert im.ndim == 3
    pil_im = Image.fromarray(im.astype(np.uint8))
  h, w = im.shape[:2]
  # print("Input shape (%d, %d)" % (h, w))
  r_h, r_w = np.float32(h) / h0, np.float32(w) / w0
  if r_h <= r_w:
    r = r_h
  else:
    r = r_w
  assert r > 0, "r = %f" % r
  new_w = int(np.round(w / r))
  new_h = int(np.round(h / r))
  assert new_w >= w0 and new_h >= h0, "(%d, %d) vs (%d, %d)" % (new_h, new_w, h0, w0)
  pil_im = pil_im.resize((new_w, new_h))
  if isinstance(x, list):
    pil_lb = pil_lb.resize((new_w, new_h))
  xstart, ystart = -1, -1
  if new_w == w0:
    xstart = 0
  else:
    xstart = int(np.random.choice(new_w - w0, 1))
  if new_h == h0:
    ystart = 0
  else:
    ystart = int(np.random.choice(new_h - h0, 1))
  im = np.array(pil_im)
  # print("Rescaled shape: ", im.shape)
  im = im[ystart:ystart+h0, xstart: xstart+w0, :]
  if isinstance(x, list):
    lb = np.array(pil_lb)
    lb = lb[ystart:ystart+h0, xstart: xstart+w0]
  return [im, lb]

def rotate(x, angle=[-45, 45], expand=True):
  """
  Rotate images (and lable-maps) at any angle
  """
  angle = np.array(angle)
  angle = np.random.randint(low=angle.min(), high=angle.max())
  if isinstance(x, list):
    assert len(x) == 2
    im, gt = x
    im = im.astype(np.uint8)
    gt = gt.astype(np.uint8)
    assert shape_match(im, gt)
    islogical = np.unique(gt).size <= 2
    # fill the rim fo the rotated image with symmetric values rather than 0
    im_rotated = transform.rotate(im, angle, mode="symmetric")
    gt_rotated = transform.rotate(gt, angle, mode="constant", preserve_range=True)
    im_rotated = np.uint8(im_rotated * 255)
    if islogical:
      gt_rotated = binary(gt_rotated)
    return [im_rotated, gt_rotated]
  else:
    assert isinstance(x, np.ndarray)
    im = im.astype(np.uint8)
    im_rotated = transform.rotate(im, angle, mode="edge")
    return im_rotated

def rotate90(x, p=[0.3, 0.4, 0.3]):
  """
  Randomly rotate image&label in 90deg
  Clockwise 90deg or 0deg or Counter-clockwise 90deg
  probabilities specified by p
  For example: p=[0.25, 0.5, 0.25]
  """
  assert isinstance(x, list) or isinstance(x, np.ndarray)
  if p != None:
    assert isinstance(p, list) or isinstance(p, np.ndarray)
    p = np.array(p)
  k = np.random.choice([-1, 0, 1], 1, p=p)
  if isinstance(x, list):
    assert len(x) == 2
    im, lb = x
    return [np.rot90(im, k=k), np.rot90(lb, k=k)]
  elif isinstance(x, np.array):
    return np.rot90(im, k=k)
  else:
    raise TypeError("Invalid type")

def crop_object_rim(x, margin=5):
  """
  NOTE!!!
  This is for saliency detection or other segmentation tasks
  Crop an image to make the object on the border of image
  """
  assert len(x) == 2
  im, gt = x
  assert np.unique(gt).size == 2, "len(np.unique(gt)) = %d" % len(np.unique(gt))
  h, w, c = im.shape
  assert im.size / gt.size == c
  [y, x] = np.where(gt != 0)
  xmin = max(x.min() - margin, 0)
  ymin = max(y.min() - margin, 0)
  xmax = min(x.max() + margin, w)
  ymax = min(y.max() + margin, h)
  if xmin == 0:
    xstart = 0
  else:
    xstart = np.random.choice(np.arange(0, xmin))
  if ymin == 0:
    ystart = 0
  else:
    ystart = np.random.choice(np.arange(0, ymin))
  if xmax == w:
    xend = w
  else:
    xend = np.random.choice(np.arange(xmax, w))
  if ymax == h:
    yend = h
  else:
    yend = np.random.choice(np.arange(ymax, h))
  return [im[ystart:yend, xstart:xend, :], gt[ystart:yend, xstart:xend]]
 

def pad_object_rim(x, margin=5):
  """
  NOTE!!!
  This is for saliency detection or other segmentation tasks
  Pad an image to make the object on the border of image
  """
  assert len(x) == 2
  im, gt = x
  assert np.unique(gt).size == 2, "len(np.unique(gt)) = %d" % len(np.unique(gt))
  h, w, c = im.shape
  assert im.size / gt.size == c
  [y, x] = np.where(gt != 0)
  xmin = max(x.min() - margin, 0)
  ymin = max(y.min() - margin, 0)
  xmax = min(x.max() + margin, w)
  ymax = min(y.max() + margin, h)
  if xmin == 0:
    wbefore = 0
  else:
    wbefore = np.random.choice(np.arange(0, xmin))
  if ymin == 0:
    hbefore = 0
  else:
    hbefore = np.random.choice(np.arange(0, ymin))
  if xmax == w:
    wafter = 0
  else:
    wafter = np.random.choice(np.arange(0, w-xmax))
  if ymax == h:
    hafter = 0
  else:
    hafter = np.random.choice(np.arange(0, h-ymax))
  impad = np.pad(im, ((hbefore, hafter), (wbefore, wafter), (0, 0)), mode="symmetric")
  if gt.ndim == 2:
    gtpad = np.pad(gt, ((hbefore, hafter), (wbefore, wafter)), mode="symmetric")
  elif gt.ndim == 3:
    gtpad = np.pad(gt, ((hbefore, hafter), (wbefore, wafter), (0, 0)), mode="symmetric")
  else:
    raise ValueError("Invalid gt shape!")
  return [impad, gtpad]

#=======================================================# 
# Operations performed on images ONLY!
#=======================================================# 
def shuffle_channel(x, p=0.5):
  """
  Random shuffle image channels
  x: ndarray with shape [h, w, channel]
  """
  assert isinstance(x, np.ndarray)
  flag = np.random.binomial(1, p)
  if x.ndim == 2 or flag == 0:
    return x
  else:
    assert x.ndim == 3
    h, w, c = x.shape
    order = np.arange(c)
    np.random.shuffle(order)
    x = x[:, :, order]
    return x

def addgaus(x, sigma=1):
  assert isinstance(x, np.ndarray)
  x = x.astype(np.float32) + np.random.rand(*x.shape) * sigma
  return x

def add_per_channel(x, sigma=5):
  assert isinstance(x, np.ndarray)
  assert x.ndim == 3
  nchannels = x.shape[2]
  noise = np.random.randint(low=-np.abs(sigma), high=np.abs(sigma), size=nchannels)
  return x + noise

def add(x, sigma=5):
  assert isinstance(x, np.ndarray)
  noise = np.random.randint(low=-np.abs(sigma), high=np.abs(sigma))
  return x + noise

def binary(x):
  assert isinstance(x, np.ndarray)
  assert x.ndim == 2 or x.ndim == 3
  x[x < (x.max() - x.min()) / 2] = 0
  x[x != 0] = 1
  return x
