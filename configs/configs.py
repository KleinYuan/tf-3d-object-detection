import numpy as np
import os

BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])
####################################################################
# Configurations for test/demo images/points cloud/calibration params
# This is the only part, you are free to change to run the demo and
# any changes in this section will not break anything
####################################################################
TEST_DATA_FP = {
    'img': '{}/example_data/1.png'.format(BASE_PATH),
    'pclds': '{}/example_data/1.bin'.format(BASE_PATH)
}

# Read https://github.com/KleinYuan/tf-object-detection#introduction
_DETECTOR_2D_MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
_DETECTOR_2D_OPTIONS = [
    'ssd_mobilenet_v1_coco_11_06_2017',
    'ssd_inception_v2_coco_11_06_2017',
    'rfcn_resnet101_coco_11_06_2017',
    'faster_rcnn_resnet101_coco_11_06_2017',
    'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
]

####################################################################
# Configurations for Main Server
# (Please read the readme in the link if you don't know what this section is:
# https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)
####################################################################

# STUB PARAM
CALIB_PARAM = {
    'P': (7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01, 0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01, 0.000000000000e+00,
          0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03),
    'Tr_velo_to_cam': (
        6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02, -1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02, 9.999753000000e-01,
        6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01),
    'R0_rect': (9.999128000000e-01, 1.009263000000e-02, -8.511932000000e-03, -1.012729000000e-02, 9.999406000000e-01, -4.037671000000e-03, 8.470675000000e-03, 4.123522000000e-03, 9.999556000000e-01)
}

####################################################################
# Configurations for BASE_SERVER Template
# (Don't touch this section if you are not fluent in tensorflow)
####################################################################

BASE_SERVER = {
    'input_tensor_names': ['image_tensor:0'],
    'output_tensor_names': ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0'],
    'device': '/gpu:0'
}

####################################################################
# Configurations or 2D Detector
# (Don't touch this section if you are not familiar with tensorflow)
####################################################################
DETECTOR_2D = {
    'MODEL_FP': '{}/pretrained/{}/frozen_inference_graph.pb'.format(BASE_PATH, _DETECTOR_2D_MODEL_NAME),
    'LABEL_FP': '{}/configs/label.pbtxt'.format(BASE_PATH),
    'NUM_CLASSES': 90,
    'FEED_IMG_SIZE': 320,
    'ONE_HOT_VECTOR_MAP': {'car': 0, 'person': 1, 'bicycle': 2}
}

####################################################################
# Configurations for 3D Detector
# (Don't touch this section if you are not familiar with Frustum PointNet)
####################################################################
DETECTOR_3D = {
    'MODEL_FP': '{}/pretrained/log_v1/model.ckpt'.format(BASE_PATH)
}

FPNET = {
    'BATCH_SIZE': 1,
    'NUM_POINT': 1024,
    'NUM_HEADING_BIN': 12,
    'NUM_SIZE_CLUSTER': 8,
    'NUM_OBJECT_POINT': 512,
    'DEVICE': '/gpu:0'
}

# FPNET labels
g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((FPNET['NUM_SIZE_CLUSTER'], 3))  # size clustrs
