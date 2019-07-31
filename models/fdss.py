import sys, os, argparse
sys.path.insert(0, 'lib')
from os.path import join, abspath, isdir
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import numpy as np
from math import ceil
parser = argparse.ArgumentParser(description='DSS')
parser.add_argument('--lossnorm', type=str, help='Normalize Loss', default="False")
parser.add_argument('--beta', type=float, help='Value of beta', default=0.8)
parser.add_argument('--aug', type=str, help='Data augmentation', default="True")
TMP_DIR = abspath('tmp')
SNAPSHOTS_DIR = abspath('snapshots')
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)
def str2bool(str1):
  if str1.lower() == 'true' or str1.lower() == '1':
    return True
  elif str1.lower() == 'false' or str1.lower() == '0':
    return False
  else:
    raise ValueError('Error!')

args = parser.parse_args()
args.lossnorm = str2bool(args.lossnorm)
args.aug = str2bool(args.aug)

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='gaussian',std=0.01), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])])
  return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv1x1(bottom, name, lr=1, wf=dict(type='gaussian',std=0.01)):
  return L.Convolution(bottom, name=name, kernel_size=1,num_output=1, weight_filler=wf,
      param=[dict(lr_mult=0.1*lr, decay_mult=1), dict(lr_mult=0.2*lr, decay_mult=0)])

def upsample(bottom, name,stride):
  s, k, pad = stride, 2 * stride, int(ceil(stride-1)/2)
  #name = "upsample%d"%s
  return L.Deconvolution(bottom, name=name, convolution_param=dict(num_output=1, 
    kernel_size=k, stride=s, pad=pad, weight_filler = dict(type="bilinear")),
      param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

def net(split):
  n = caffe.NetSpec()
  if split=='train':
    data_params = dict(mean=(104.00699, 116.66877, 122.67892))
    data_params['root'] = './data/MSRA-B/'
    data_params['source'] = "train_list.txt"
    data_params['shuffle'] = True
    data_params['aug'] = args.aug
    data_params['ignore_label'] = -1 # ignore label
    n.data, n.label = L.Python(module='pylayer', layer='ImageLabelmapDataLayer', ntop=2, \
    param_str=str(data_params))
    loss_param = dict(normalize=args.lossnorm)
    if data_params.has_key('ignore_label'):
      loss_param['ignore_label'] = data_params['ignore_label']
  elif split == 'test':
    n.data = L.Input(name = 'data', input_param=dict(shape=dict(dim=[1,3,500,500])))
  else:
    raise Exception("Invalid phase")
  
  n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=5)
  n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
  n.pool1 = max_pool(n.relu1_2)

  n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
  n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
  n.pool2 = max_pool(n.relu2_2)

  n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
  n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
  n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
  n.pool3 = max_pool(n.relu3_3)

  n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
  n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
  n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
  n.pool4 = max_pool(n.relu4_3)
  
  n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
  n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
  n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
  n.pool5 = max_pool(n.relu5_3)
  n.pool5a = L.Pooling(n.pool5, pool=P.Pooling.AVE, kernel_size=3, stride=1,pad=1)
  ###DSN conv 6###
  n.conv1_dsn6,n.relu1_dsn6=conv_relu(n.pool5a,512,ks=7, pad=3)
  n.conv2_dsn6,n.relu2_dsn6=conv_relu(n.relu1_dsn6,512,ks=7, pad=3)
  n.conv3_dsn6=conv1x1(n.relu2_dsn6, 'conv3_dsn6')
  n.score_dsn6_up = upsample(n.conv3_dsn6, stride=32,name='upsample32_in_dsn6')
  n.upscore_dsn6 = crop(n.score_dsn6_up, n.data)
  if split=='train':
    n.sigmoid_dsn6 = L.Sigmoid(n.upscore_dsn6)
    floss_param = dict()
    floss_param['name']='dsn6'
    floss_param['beta']=args.beta
    n.loss_dsn6 = L.Python(n.sigmoid_dsn6,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_dsn6 = L.Sigmoid(n.upscore_dsn6)
  ###DSN conv 5###
  n.conv1_dsn5,n.relu1_dsn5=conv_relu(n.conv5_3,512,ks=5, pad=2)
  n.conv2_dsn5,n.relu2_dsn5=conv_relu(n.relu1_dsn5,512,ks=5, pad=2)
  n.conv3_dsn5=conv1x1(n.relu2_dsn5, 'conv3_dsn5')
  n.score_dsn5_up = upsample(n.conv3_dsn5, stride=16,name='upsample16_in_dsn5')
  n.upscore_dsn5 = crop(n.score_dsn5_up, n.data)
  if split=='train':
    n.sigmoid_dsn5 = L.Sigmoid(n.upscore_dsn5)
    floss_param['name']='dsn5'
    floss_param['beta']=args.beta
    n.loss_dsn5 = L.Python(n.sigmoid_dsn5,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_dsn5 = L.Sigmoid(n.upscore_dsn5)
  ###DSN conv 4###
  n.conv1_dsn4,n.relu1_dsn4=conv_relu(n.conv4_3,256,ks=5, pad=2)
  n.conv2_dsn4,n.relu2_dsn4=conv_relu(n.relu1_dsn4,256,ks=5, pad=2)
  n.conv3_dsn4=conv1x1(n.relu2_dsn4, 'conv3_dsn4')

  n.score_dsn6_up_4  = upsample(n.conv3_dsn6, stride=4,name='upsample4_dsn6')
  n.upscore_dsn6_4 = crop(n.score_dsn6_up_4, n.conv3_dsn4)
  n.score_dsn5_up_4  = upsample(n.conv3_dsn5, stride=2,name='upsample2_dsn5')
  n.upscore_dsn5_4 = crop(n.score_dsn5_up_4, n.conv3_dsn4)
  n.concat_dsn4 = L.Eltwise(n.conv3_dsn4,
                      n.upscore_dsn6_4,
                      n.upscore_dsn5_4,
                      name="concat_dsn4")
  n.conv4_dsn4=conv1x1(n.concat_dsn4, 'conv4_dsn4')
  n.score_dsn4_up = upsample(n.conv4_dsn4, stride=8,name='upsample8_in_dsn4')
  n.upscore_dsn4 = crop(n.score_dsn4_up, n.data)
  if split=='train':
    n.sigmoid_dsn4 = L.Sigmoid(n.upscore_dsn4)
    floss_param['name']='dsn4'
    floss_param['beta']=args.beta
    n.loss_dsn4 = L.Python(n.sigmoid_dsn4,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_dsn4 = L.Sigmoid(n.upscore_dsn4)
  ### DSN conv 3 ###
  n.conv1_dsn3,n.relu1_dsn3=conv_relu(n.conv3_3,256,ks=5, pad=2)
  n.conv2_dsn3,n.relu2_dsn3=conv_relu(n.relu1_dsn3,256,ks=5, pad=2)
  n.conv3_dsn3=conv1x1(n.relu2_dsn3, 'conv3_dsn3')

  n.score_dsn6_up_3  = upsample(n.conv3_dsn6, stride=8,name='upsample8_dsn6')
  n.upscore_dsn6_3 = crop(n.score_dsn6_up_3, n.conv3_dsn3)
  n.score_dsn5_up_3  = upsample(n.conv3_dsn5, stride=4,name='upsample4_dsn5')
  n.upscore_dsn5_3 = crop(n.score_dsn5_up_3, n.conv3_dsn3)
  n.concat_dsn3 = L.Eltwise(n.conv3_dsn3,
                      n.upscore_dsn6_3,
                      n.upscore_dsn5_3,
                      name='concat')
  n.conv4_dsn3=conv1x1(n.concat_dsn3, 'conv4_dsn3')
  n.score_dsn3_up = upsample(n.conv4_dsn3, stride=4,name='upsample4_in_dsn3')
  n.upscore_dsn3 = crop(n.score_dsn3_up, n.data)
  if split=='train':
    n.sigmoid_dsn3 = L.Sigmoid(n.upscore_dsn3)
    floss_param['name']='dsn3'
    floss_param['beta']=args.beta
    n.loss_dsn3 = L.Python(n.sigmoid_dsn3,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_dsn3 = L.Sigmoid(n.upscore_dsn3)
  ### DSN conv 2 ###
  n.conv1_dsn2,n.relu1_dsn2=conv_relu(n.conv2_2,128,ks=3, pad=1)
  n.conv2_dsn2,n.relu2_dsn2=conv_relu(n.relu1_dsn2,128,ks=3, pad=1)
  n.conv3_dsn2=conv1x1(n.relu2_dsn2, 'conv3_dsn2')

  n.score_dsn6_up_2  = upsample(n.conv3_dsn6, stride=16,name='upsample16_dsn6')
  n.upscore_dsn6_2 = crop(n.score_dsn6_up_2, n.conv3_dsn2)
  n.score_dsn5_up_2  = upsample(n.conv3_dsn5, stride=8,name='upsample8_dsn5')
  n.upscore_dsn5_2 = crop(n.score_dsn5_up_2, n.conv3_dsn2)
  n.score_dsn4_up_2  = upsample(n.conv4_dsn4, stride=4,name='upsample4_dsn4')
  n.upscore_dsn4_2 = crop(n.score_dsn4_up_2, n.conv3_dsn2)
  n.score_dsn3_up_2  = upsample(n.conv4_dsn3, stride=2,name='upsample2_dsn3')
  n.upscore_dsn3_2 = crop(n.score_dsn3_up_2, n.conv3_dsn2)
  n.concat_dsn2 = L.Eltwise(n.conv3_dsn2,
                      n.upscore_dsn5_2,
                      n.upscore_dsn4_2,
                      n.upscore_dsn6_2,
                      n.upscore_dsn3_2,
                      name='concat')
  n.conv4_dsn2=conv1x1(n.concat_dsn2, 'conv4_dsn2')
  n.score_dsn2_up = upsample(n.conv4_dsn2, stride=2,name='upsample2_in_dsn2')
  n.upscore_dsn2 = crop(n.score_dsn2_up, n.data)
  if split=='train':
    n.sigmoid_dsn2 = L.Sigmoid(n.upscore_dsn2)
    floss_param['name']='dsn2'
    floss_param['beta']=args.beta
    n.loss_dsn2 = L.Python(n.sigmoid_dsn2,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_dsn2 = L.Sigmoid(n.upscore_dsn2)
  ## DSN conv 1 ###
  n.conv1_dsn1,n.relu1_dsn1=conv_relu(n.conv1_2,128,ks=3, pad=1)
  n.conv2_dsn1,n.relu2_dsn1=conv_relu(n.relu1_dsn1,128,ks=3, pad=1)
  n.conv3_dsn1=conv1x1(n.relu2_dsn1, 'conv3_dsn1')

  n.score_dsn6_up_1  = upsample(n.conv3_dsn6, stride=32,name='upsample32_dsn6')
  n.upscore_dsn6_1 = crop(n.score_dsn6_up_1, n.conv3_dsn1)
  n.score_dsn5_up_1  = upsample(n.conv3_dsn5, stride=16,name='upsample16_dsn5')
  n.upscore_dsn5_1 = crop(n.score_dsn5_up_1, n.conv3_dsn1)
  n.score_dsn4_up_1  = upsample(n.conv4_dsn4, stride=8,name='upsample8_dsn4')
  n.upscore_dsn4_1 = crop(n.score_dsn4_up_1, n.conv3_dsn1)
  n.score_dsn3_up_1  = upsample(n.conv4_dsn3, stride=4,name='upsample4_dsn3')
  n.upscore_dsn3_1 = crop(n.score_dsn3_up_1, n.conv3_dsn1)

  n.concat_dsn1 = L.Eltwise(n.conv3_dsn1,
                      n.upscore_dsn5_1,
                      n.upscore_dsn4_1,
                      n.upscore_dsn6_1,
                      n.upscore_dsn3_1,
                      name='concat')
  n.score_dsn1_up=conv1x1(n.concat_dsn1, 'conv4_dsn1')
  n.upscore_dsn1 = crop(n.score_dsn1_up, n.data)
  if split=='train':
    n.sigmoid_dsn1 = L.Sigmoid(n.upscore_dsn1)
    floss_param['name']='dsn1'
    floss_param['beta']=args.beta
    n.loss_dsn1 = L.Python(n.sigmoid_dsn1,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_dsn1 = L.Sigmoid(n.upscore_dsn1)
  ### Eltwise and multiscale weight layer ###
  n.concat_upscore = L.Eltwise(n.upscore_dsn1,
                      n.upscore_dsn2,
                      n.upscore_dsn3,
                      n.upscore_dsn4,
                      n.upscore_dsn5,
                      n.upscore_dsn6,
                      name='concat')
  n.upscore_fuse=conv1x1(n.concat_upscore, 'new_score_weighting', wf=dict({'type': 'constant', 'value':np.float(1)/6 }))
  if split=='train':
    n.sigmoid_fuse = L.Sigmoid(n.upscore_fuse)
    floss_param['name']='fuse'
    floss_param['beta']=args.beta
    n.loss_fuse = L.Python(n.sigmoid_fuse,n.label,module='floss', layer='FmeasureLossLayer',param_str=str(floss_param),ntop=1,loss_weight=1)
  else:
    n.sigmoid_fuse = L.Sigmoid(n.upscore_fuse)
  return n.to_proto()

pt_filename = join(TMP_DIR, 'fdss')
snapshot_filename = join(SNAPSHOTS_DIR, 'fdss')
pt_filename = "%s_beta%.2f" % (pt_filename, args.beta)
snapshot_filename = "%s_beta%.2f" % (snapshot_filename, args.beta)

if args.lossnorm:
  pt_filename += "_lossnorm"
  snapshot_filename += "_lossnorm"
if args.aug:
  pt_filename += "_aug"
  snapshot_filename += "_aug"

print("%s\n%s" % (pt_filename, snapshot_filename))
def make_net():
  with open('%s_train.pt' %(pt_filename), 'w') as f:
    f.write(str(net('train')))
  with open('%s_test.pt' %(pt_filename), 'w') as f:
    f.write(str(net('test')))
def make_solver():
  sp = {}
  sp['net'] = '"%s_train.pt"' %(pt_filename)
  if args.lossnorm:
    sp['base_lr'] = '1e-3'
  else:
    sp['base_lr'] = '1e-3'
  sp['lr_policy'] = '"step"'
  sp['momentum'] = '0.9'
  sp['weight_decay'] = '0.0001'
  sp['iter_size'] = '10'
  sp['stepsize'] = '5000'
  sp['display'] = '10'
  sp['snapshot'] = '2000'
  sp['snapshot_prefix'] = '"%s"' % snapshot_filename
  sp['gamma'] = '0.1'
  sp['max_iter'] = '40000'
  sp['solver_mode'] = 'GPU'
  f = open('%s_solver.pt' % pt_filename, 'w')
  for k, v in sorted(sp.items()):
      if not(type(v) is str):
          raise TypeError('All solver parameters must be strings')
      f.write('%s: %s\n'%(k, v))
  f.close()

def make_all():
  make_net()
  make_solver()

if __name__ == '__main__':
  make_all()
