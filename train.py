from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys, os, argparse
from scipy.io import savemat
import datetime
sys.path.insert(0, 'lib')
from os.path import isfile, join, isdir, abspath
import cv2
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
parser = argparse.ArgumentParser(description='Training DSS.')
parser.add_argument('--gpu', type=int, help='gpu ID', default=0)
parser.add_argument('--solver', type=str, help='solver', default='models/floss_solver.prototxt')
parser.add_argument('--weights', type=str, help='base model', default='models/vgg16convs.caffemodel')
parser.add_argument('--debug', type=str, help='debug mode', default='False')
def str2bool(str1):
  if "true" in str1.lower() or "1" in str1.lower():
    return True
  elif "false" in str1.lower() or "0" in str1.lower():
    return False
args = parser.parse_args()
assert isfile(args.solver)
assert isfile(args.weights)
DEBUG = str2bool(args.debug)
CACHE_FREQ = 1
CACHE_DIR = abspath('data/cache')
if not isdir(CACHE_DIR):
  os.makedirs(CACHE_DIR)
if DEBUG:
  from pytools.image import overlay
  from pytools.misc import blob2im
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
def interp_surgery(net, layers):
  for l in layers:
    m, k, h, w = net.params[l][0].data.shape
    if m != k:
      print('input + output channels need to be the same')
      raise
    if h != w:
      print('filters need to be square')
      raise
    filt = upsample_filt(h)
    net.params[l][0].data[range(m), range(k), :, :] = filt
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
if not isdir('snapshots'):
  os.makedirs('snapshots')
solver = caffe.SGDSolver(args.solver)
# get snapshot_prefix
solver_param = caffe_pb2.SolverParameter()
with open(args.solver, 'rb') as f:
  text_format.Merge(f.read(), solver_param)
max_iter = solver_param.max_iter
# net surgery
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)
solver.net.copy_from(args.weights)
for p in solver.net.params:
  param = solver.net.params[p]
  for i in range(len(param)):
    print(p, "param[%d]: mean=%.5f, std=%.5f"%(i, solver.net.params[p][i].data.mean(), \
    solver.net.params[p][i].data.std()))
if DEBUG:
  now = datetime.datetime.now()
  cache_dir = join(CACHE_DIR, "%s-%s-%dH-%dM-%dS" % (args.solver.split(os.sep)[-1], str(now.date()), now.hour, now.minute,
  now.second))
  if not isdir(cache_dir):
    os.makedirs(cache_dir)
  for i in range(1, max_iter + 1, CACHE_FREQ):
    cache_fn = join(cache_dir, "iter%d" % i)
    solver.step(CACHE_FREQ)
    keys = [None] * 7
    for i in range(len(keys)):
      if i <= 5:
        keys[i] = "sigmoid_dsn%d" % (i + 1)
      else:
        keys[i] = "sigmoid_fuse"
    mat_dict = dict()
    for k in keys:
      mat_dict[k + "_data"] = np.squeeze(solver.net.blobs[k].data)
      mat_dict[k + "_grad"] = np.squeeze(solver.net.blobs[k].diff)
    im = blob2im(solver.net.blobs['data'].data)
    mat_dict["image"] = im
    lb = np.squeeze(solver.net.blobs['label'].data)
    mat_dict["label"] = lb
    savemat(cache_fn, mat_dict)
    im = overlay(im, lb)
    dsn1 = np.squeeze(solver.net.blobs['sigmoid_dsn1'].data)
    dsn2 = np.squeeze(solver.net.blobs['sigmoid_dsn2'].data)
    dsn3 = np.squeeze(solver.net.blobs['sigmoid_dsn3'].data)
    dsn4 = np.squeeze(solver.net.blobs['sigmoid_dsn4'].data)
    dsn5 = np.squeeze(solver.net.blobs['sigmoid_dsn5'].data)
    dsn6 = np.squeeze(solver.net.blobs['sigmoid_dsn6'].data)
    fuse = np.squeeze(solver.net.blobs['sigmoid_fuse'].data)
    dss_fuse = (dsn3 + dsn4 + dsn5 + fuse) / 4
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes[0, 0].imshow(im)
    axes[0, 0].set_title("image and label")
    axes[0, 1].imshow(dsn1, cmap=cm.Greys_r)
    axes[0, 1].set_title("DSN1")
    axes[0, 2].imshow(dsn2, cmap=cm.Greys_r)
    axes[0, 2].set_title("DSN2")
    axes[1, 0].imshow(dsn3, cmap=cm.Greys_r)
    axes[1, 0].set_title("DSN3")
    axes[1, 1].imshow(dsn4, cmap=cm.Greys_r)
    axes[1, 1].set_title("DSN4")
    axes[1, 2].imshow(dsn5, cmap=cm.Greys_r)
    axes[1, 2].set_title("DSN5")
    axes[2, 0].imshow(dsn6, cmap=cm.Greys_r)
    axes[2, 0].set_title("DSN6")
    axes[2, 1].imshow(fuse, cmap=cm.Greys_r)
    axes[2, 1].set_title("fuse (dsn1~6)")
    axes[2, 2].imshow(dss_fuse, cmap=cm.Greys_r)
    axes[2, 2].set_title("DSS style fuse (dsn3~5 + fuse)")
    plt.savefig(cache_fn+'.jpg')
    plt.close(fig)
    print("Saving cache file to %s" % cache_fn)
else:
  solver.solve()
