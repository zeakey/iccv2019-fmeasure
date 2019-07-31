

<h1 align="center">Optimizing the F-measure for Threshold-free Salient Object Detection</h1>

Code accompanying the paper **Optimizing the F-measure for Threshold-free Salient Object Detection**.

<div width=280px align="center">
<a href="http://kaizhao.net/fmeasure">
<img src="http://data.kaizhao.net/projects/fmeasure-saliency/qr-code.png" width=150px>
</a>
<a href="http://data.kaizhao.net/publications/iccv2019fmeasure.pdf">
<img src="http://data.kaizhao.net/projects/fmeasure-saliency/paper-thumbnail.png" width=130px>
</a>
</div>

## Howto
1. Download and build [caffe](https://github.com/bvlc/caffe) with python interface;
2. Download the MSRA-B dataset to `data/` and the initial [VGG weights](http://data.kaizhao.net/projects/fmeasure-saliency/vgg16convs.caffemodel) to `model/`
3. Generate network and solver prototxt via `python model/fdss.py`;
4. Start training the DSS+FLoss model with `python train.py --solver tmp/fdss_beta0.80_aug_solver.pt`

## Loss surface
The proposed FLoss holds considerable gradients even in the saturated
area, resulting in polarized predictions that are stable against the threshold.

<p align="center">
  <img src="http://data.kaizhao.net/projects/fmeasure-saliency/loss-surface.svg" width=100%>
</p>
<p align="center">
Loss surface of FLoss (left), Log-FLoss (mid) and Cross-entropy loss (right). FLoss holds larger gradients in the saturated
area, leading to high-contrast predictions.
</p>

## Example detection results
<p align="center">
<img src="http://data.kaizhao.net/projects/fmeasure-saliency/example-detections.png" width=800px>
</p>
<p align="center">
Several detection results. Our method results in high-contrast detections.
</p>

## Stability against threshold
<p align="center">
<img src="http://data.kaizhao.net/projects/fmeasure-saliency/f-thres.svg" width=400px>
</p>
<p align="center">
FLoss (solid lines) achieves high F-measure under a larger range
of thresholds, presenting stability against the changing of threshold.
</p>

## Pretrained models

For pretrained models and evaluation results, please visit <http://kaizhao.net/fmeasure>.

___
If you have any problem using this code, please contact [Kai Zhao](http://kaizhao.net).


