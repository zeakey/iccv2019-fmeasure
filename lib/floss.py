import caffe
import numpy as np
from numpy import logical_and as land, logical_or as lor, logical_not as lnot
import caffe
FLT_MIN=1e-16

class FmeasureLossLayer(caffe.Layer):
    def setup(self, bottom, top):
      if len(bottom) != 2:
          raise Exception("Need two inputs to compute distance.")
      params = eval(self.param_str)
      self.log = False
      if 'log' in params:
        self.log = bool(params['log'])
      self.counter=0
      self.beta = np.float(params['beta'])
      self.DEBUG = True

    def reshape(self, bottom, top):
      if bottom[0].count != bottom[1].count:
          raise Exception("Inputs must have the same dimension.")
      self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
      top[0].reshape(1)

    def forward(self, bottom, top):
      """
      F = \frac{(1+\beta)pr}{\beta p + r}
      loss = 1 - F
      p = \frac{TP}{TP + FP}
      r = \frac{TP}{TP + FN}
      See http://kaizhao.net/fmeasure
      """
      pred = np.squeeze(bottom[0].data[...])
      target = np.squeeze(bottom[1].data[...])
      target = target > 0
      h, w = target.shape
      assert pred.max() <= 1 and pred.min() >= 0, "pred.max = %f, pred.min = %f" % (pred.max(), pred.min())
      self.TP=np.sum(target * pred)
      self.H = self.beta * target.sum() + pred.sum()
      self.fmeasure = (1 + self.beta) * self.TP / (self.H + FLT_MIN)
      if self.log:
        # loss = -\log{F-measure}
        loss = -np.log(self.fmeasure + FLT_MIN)
      else:
        # loss = 1 - F-measure
        loss = 1 - self.fmeasure
      top[0].data[0] = loss

    def backward(self, top, propagate_down, bottom):
      """
      grad[i] = \frac{(1+\beta)TP}{H^2} - \frac{(1+\beta)y_i}{H}
      See http://kaizhao.net/fmeasure
      """
      pred = bottom[0].data[...]
      target = bottom[1].data[...]
      grad = (1 + self.beta) * self.TP / (self.H**2 + FLT_MIN) - \
                    (1+self.beta) * target / (self.H + FLT_MIN)
      if self.log:
        grad /= (self.fmeasure + FLT_MIN)
      bottom[0].diff[...] = grad
