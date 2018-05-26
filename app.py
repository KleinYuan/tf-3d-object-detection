import sys
import cv2
import numpy as np
import tensorflow as tf
import cPickle as pickle


class Net(object):
	graph = None
	image_tensor = None
	boxes = None
	scores = None
	classes = None
	num_detections = None
	in_progress = False
	session = None

	pc_tr = None
	one_hot_vector_tr = None
	is_training_tr = None
	stage1_center_tr = None
	center_tr = None
	side_residuals_tr = None
	size_residuals_normalized_tr = None
	mask_tr = None
	heading_residuals_normalized_tr = None
	heading_scores_tr = None
	heading_residuals_tr = None
	size_scores_tr = None
	center_boxnet_tr = None
	mask_logits_tr = None

	def __init__(self, graph_fp):
		self.graph_fp = graph_fp
		self._load_graph()
		self._init_predictor()

	def _load_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		tf.get_default_graph().finalize()

	# ('stage1_center', <tf.Tensor 'add:0' shape=(32, 3) dtype=float32>)
	# ('center', <tf.Tensor 'add_1:0' shape=(32, 3) dtype=float32>)
	# ('size_residuals', <tf.Tensor 'mul_2:0' shape=(32, 8, 3) dtype=float32>)
	# ('size_residuals_normalized', <tf.Tensor 'Reshape:0' shape=(32, 8, 3) dtype=float32>)
	# ('mask', <tf.Tensor 'Squeeze_1:0' shape=(32, 1024) dtype=float32>)
	# ('heading_residuals_normalized', <tf.Tensor 'Slice_5:0' shape=(32, 12) dtype=float32>)
	# ('heading_scores', <tf.Tensor 'Slice_4:0' shape=(32, 12) dtype=float32>)
	# ('heading_residuals', <tf.Tensor 'mul_1:0' shape=(32, 12) dtype=float32>)
	# ('size_scores', <tf.Tensor 'Slice_6:0' shape=(32, 8) dtype=float32>)
	# ('center_boxnet', <tf.Tensor 'Slice_3:0' shape=(32, 3) dtype=float32>)
	# ('mask_logits', <tf.Tensor 'Squeeze:0' shape=(32, 1024, 2) dtype=float32>)

	def _init_predictor(self):
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True
		with self.graph.as_default():
			self.session = tf.Session(config=tf_config, graph=self.graph)

			self.pc_tr = self.graph.get_tensor_by_name('Placeholder:0')
			self.one_hot_vector_tr = self.graph.get_tensor_by_name('Placeholder_1:0')
			self.is_training_tr = self.graph.get_tensor_by_name('Placeholder_8:0')
			self.stage1_center_tr = self.graph.get_tensor_by_name('add:0')
			self.center_tr = self.graph.get_tensor_by_name('add_1:0')
			self.side_residuals_tr = self.graph.get_tensor_by_name('mul_2:0')
			self.size_residuals_normalized_tr = self.graph.get_tensor_by_name('Reshape:0')
			self.mask_tr = self.graph.get_tensor_by_name('Squeeze_1:0')
			self.heading_residuals_normalized_tr = self.graph.get_tensor_by_name('Slice_5:0')
			self.heading_scores_tr = self.graph.get_tensor_by_name('Slice_4:0')
			self.heading_residuals_tr = self.graph.get_tensor_by_name('mul_1:0')
			self.size_scores_tr = self.graph.get_tensor_by_name('Slice_6:0')
			self.center_boxnet_tr = self.graph.get_tensor_by_name('Slice_3:0')
			self.mask_logits_tr = self.graph.get_tensor_by_name('Squeeze:0')

	def predict(self, pc, one_hot_vector, is_training):
		self.in_progress = True

		with self.graph.as_default():
			feed_dict = {
				self.pc_tr: pc,
				self.one_hot_vector_tr: one_hot_vector,
				self.is_training_tr: is_training
			}
			(_, _center, _, _, _mask, _, _, _, _, _, _mask_logits) = self.session.run(
				[self.stage1_center_tr,
				 self.center_tr,
				 self.side_residuals_tr,
				 self.size_residuals_normalized_tr,
				 self.mask_tr,
				 self.heading_residuals_normalized_tr,
				 self.heading_scores_tr,
				 self.heading_residuals_tr,
				 self.size_scores_tr,
				 self.center_boxnet_tr,
				 self.mask_logits_tr
				 ],
				feed_dict=feed_dict)

		self.in_progress = False

	def get_status(self):
		return self.in_progress

	def kill_predictor(self):
		self.session.close()
		self.session = None


# with open(r"frustum_carpedcyc_val_rgb_detection.pickle", "rb") as input_file:
# 	content = pickle.load(input_file)
# 	print len(content)
# 	print type(content)
# 	print content[0:200]

net = Net(graph_fp='fpnetv1.pb')
_pc = np.ones((32, 1024, 4))
_one_hot_vector = np.ones((32, 3))
_is_training = True
net.predict(_pc, _one_hot_vector, _is_training)
