# Summary

![3d](https://user-images.githubusercontent.com/8921629/41188550-0ed19016-6b74-11e8-92fb-193a8160d0e2.png)

(Below is from a data in [KITTI 3D Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d))

![semi-endtoend](https://user-images.githubusercontent.com/8921629/41068890-76807090-69a0-11e8-9794-62fc394667b3.png)

# Run demo

#### 1. Requirements

- [X] MacOS or Ubuntu

- [X] Tensorflow

- [X] Mayavi (visualization Only)

- [X] OpenCV

- [ ] Anaconda preferred (optional)

### 2. Clone this repo

```
git clone https://github.com/KleinYuan/tf-3d-object-detection.git
```

### 2. Install Dependencies

```
# Simply run this in this project root folder
cd tf-3d-object-detection
pip install -r requirements.txt
```

If you meet error install say `opencv`, do `conda install opencv` if you use Anaconda. Otherwise, dude, build from source and let's call it a day.

### 3. Pick a 2D Object Detection Model

In here we support 5 different 2D Detection models:

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) | fast | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) | fast | 24 | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)  | medium | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) | medium | 32 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) | slow | 37 | Boxes |

Pick one of those that makes you feel good, and find it in the list -- [`_DETECTOR_2D_OPTIONS` in `configs/configs`](https://github.com/KleinYuan/tf-3d-object-detection/blob/master/configs/configs.py#L17),
then replace it with the value of [`_DETECTOR_2D_MODEL_NAME`](https://github.com/KleinYuan/tf-3d-object-detection/blob/master/configs/configs.py#L16).

And by default, I use [`ssd_mobilenet_v1_coco_11_06_2017`](https://github.com/KleinYuan/tf-3d-object-detection/blob/master/configs/configs.py#L16) due to it's fast.

### 4. Download Test Data

Due to the license of KITTI is waaaaaaaaaaay to long to read, I will just tell ya how to do it instead of running a risk to attach here with some data from KITTI, which
when I downloaded it I clicked some button to have agreed on something that's TLTR.

```
# Step1 Go to http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# Step2 Do "Download left color images of object data set (12 GB)"
# Step3 Do "Download Velodyne point clouds, if you want to use laser information (29 GB)"
# Step4 Do "Download camera calibration matrices of object data set (16 MB)"
# Step5 Unzip all those three zip files and you will find ~7000ish training datasets, each pair include velodyne, image and calibration
# Step6 Pick one of them, copy and paste it under example_data folder, and name the image to 1.png, and velodyne file to 1.bin
# Step7 Open calibration file, find corresponding item and replace it with CALIB_PARAM in configs/configs.py, by default, it's from training/000000.txt
# Step8 Really sorry to let you go thru last 7 Steps and I think I may come up with a better idea to do it with one button
```

### 5. Download Pretrained Model

As you may see, this project combined 2 Deep Neural Networks together. Therefore, yes you need to download two pre-trained model.

| 2D Object Detector Model  | 3D Object Detector Model |
| ------------ | :--------------: |
| [Download Link](https://github.com/KleinYuan/tf-object-detection/blob/master/README.md#introduction)| [Download v1 and v2 is not supported yet](https://shapenet.cs.stanford.edu/media/frustum_pointnets_snapshots.zip) (originally from [here](https://github.com/Dark-Rinnegan/frustum-pointnets/tree/app#training-frustum-pointnets))|

Then, unzip them and put them under [`pretrained`](https://github.com/KleinYuan/tf-3d-object-detection/tree/master/pretrained) folder. Also, renamed the `checkpoint.txt` file to `checkpoint` even though it's useless and you cannot freeze it :unhappy: .


The folder will look like this:

```
--tf-3d-object-detection
  |-- pretrained
      |--log_v1
          |-- checkpoint (originally named checkpoint.txt)
          |-- log_train.txt
          |-- model.ckpt.data-00000-of-00001
          |-- model.ckpt
          |-- model.ckpt.meta
      |-- ssd_mobilenet_v1_coco_11_06_2017 (or other names if you decide to use different ones)
          |-- frozen_inference_graph.pb
          |-- graph.pbtxt
          |-- model.ckpt-0.data-00000-of-00001
          |-- model.ckpt-0.index
          |-- model.ckpt-0.meta

```

You may realize this [fact](https://github.com/Dark-Rinnegan/frustum-pointnets/tree/app/app#intro) thus 3D object detection model is not really frozenable one.

(Hopefully they can disclose the original tensorflow ops for v1 so that we can remove both `tf.py_func` and freeze the model)

### 6. Run Demo

```
# If you use Pycharm, just click the green run button
# If not, navigate to root folder of this repo and run:
python apps/demo.py

# If it complains, yo, I cannot find some modules, yo, do:
export PYTHONPATH='.'
python apps/demo.py

# And if you still have the issue, man, you must really mess up with your python env.
# I don't wanan help you on that in this readme and don't create an issue for that as well.
# You shall either try using anaconda or find a python knower to help you with it
# Or, just do STACKOVERFLOW like other pals do

```

Then you should be able to see 3 Windows pop up in order, and don't forget to `Press any key to continue` as the terminal mention.


# References

- [X] Project Template: [AIInAi/tensorflow-project-template](https://github.com/AIInAi/tensorflow-project-template)

- [X] FPNet Code: [Dark-Rinnegan/frustum-pointnets](https://github.com/Dark-Rinnegan/frustum-pointnets/tree/app/app)
