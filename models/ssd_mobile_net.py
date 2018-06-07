import base_server


class SSDMobileNet(base_server.BaseServer):

	def get_model_name(self):
		return 'SSDMobileNet based on Base server'

	def inference_vebose(self, data):
		self.inference(data)
		# TODO: Calculate  one_hot_vectors and return together with bboxes_2d
		one_hot_vectors = []
		bboxes_2d = []
		return bboxes_2d, one_hot_vectors
