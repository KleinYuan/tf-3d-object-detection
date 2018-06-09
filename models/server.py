import frustum_point_net, frustum_proposal, detector_2d
import numpy as np
from configs import configs
from utils import utils


class Server(object):

	frt_proposal_server = None
	detector_2d = None
	detector_3d = None
	in_progress = False
	CALIB_PARAM = configs.CALIB_PARAM
	NUM_POINT = configs.FPNET['NUM_POINT']
	DETECTOR_3D_MODEL_FP = configs.DETECTOR_3D_MODEL_FP
	NUM_HEADING_BIN = configs.FPNET['NUM_HEADING_BIN']
	DETECTOR_2D_MODEL_FP = configs.DETECTOR_2D_MODEL_FP
	input_tensor_names = configs.BASE_SERVER['input_tensor_names']
	output_tensor_names = configs.BASE_SERVER['output_tensor_names']
	device = configs.BASE_SERVER['device']

	def __init__(self):
		self._load_params()
		self._init_detector_2d()
		self._init_frt_proposal_server()
		self._init_detector_3d()

	def _load_params(self):
		print('[Server] Init Params ...')
		self.calib_param = self.CALIB_PARAM

	def _init_frt_proposal_server(self):
		print('[Server] Init frustum proposal server ...')
		self.frt_proposal_server = frustum_proposal.FrustumProposal(self.calib_param)

	def _init_detector_2d(self):
		print('[Server] Init image 2d detection server ...')
		self.detector_2d = detector_2d.Detector2D(
			model_fp=self.DETECTOR_2D_MODEL_FP,
			input_tensor_names=self.input_tensor_names,
			output_tensor_names=self.output_tensor_names,
			device=self.device)

	def _init_detector_3d(self):
		print('[Server] Init 3d object detection server ...')
		self.detector_3d = frustum_point_net.FPNetPredictor(model_fp=self.DETECTOR_3D_MODEL_FP)

	def predict(self, inputs):
		print('[Server | Init] Run prediction ...')
		# Process one image and one frame of point cloud at once
		assert 'img' and 'pclds' in inputs
		self.in_progress = True

		print('[Server | Step1] Run 2d bounding box detection ...')
		bboxes_2d, one_hot_vectors = self.detector_2d.inference_verbose(inputs['img'])

		print('[Server | Step2] Run frustum proposal server ...')
		f_prop_cam_all, f_prop_velo_all = self.frt_proposal_server.get_frustum_proposal(inputs['img'].shape, bboxes_2d, inputs['pclds'])

		print('[Server | Step3] Down sampling points ...')
		for idx, f_prop_cam in enumerate(f_prop_cam_all):
			choice = np.random.choice(f_prop_cam.shape[0], self.NUM_POINT, replace=True)
			f_prop_cam_all[idx] = f_prop_cam[choice, :]

		print('[Server | Step4] Detetcing 3D Bounding boxes from frustum proposals ...')
		logits, centers, \
		heading_logits, heading_residuals, \
		size_scores, size_residuals = self.detector_3d.predict(pc=f_prop_cam_all, one_hot_vec=one_hot_vectors)

		print('[Server | Step5] Preparing visualization ...')
		for idx in range(len(centers)):

			heading_class = np.argmax(heading_logits, 1)
			size_logits = size_scores
			size_class = np.argmax(size_logits, 1)
			size_residual = np.vstack([size_residuals[0, size_class[idx], :]])
			heading_residual = np.array([heading_residuals[idx, heading_class[idx]]])  # B,
			heading_angle = utils.class2angle(heading_class[idx], heading_residual[idx], self.NUM_HEADING_BIN)
			box_size = utils.class2size(size_class[idx], size_residual[idx])
			corners_3d = utils.get_3d_box(box_size, heading_angle, centers[idx])

			corners_3d_in_velo_frame = np.zeros_like(corners_3d)
			centers_in_velo_frame = np.zeros_like(centers)
			corners_3d_in_velo_frame[:, 0:3] = self.frt_proposal_server.project_rect_to_velo(corners_3d[:, 0:3])
			centers_in_velo_frame[:, 0:3] = self.frt_proposal_server.project_rect_to_velo(centers[:, 0:3])
			utils.viz_single(f_prop_velo_all[idx])
			utils.viz(f_prop_velo_all[idx], centers_in_velo_frame, corners_3d_in_velo_frame, inputs['pclds'])

		self.in_progress = False
