import sys
import _base_server
import cv2
import numpy as np
from configs import configs

sys.path.append("..")
import libs.label_map_util


class Detector2D(_base_server.BaseServer):

    img_height_received = 0
    img_width_received = 0
    img_feed = None
    img_received = None
    img_resized = None
    num_classes = configs.DETECTOR_2D['NUM_CLASSES']
    img_resize_size = configs.DETECTOR_2D['FEED_IMG_SIZE']
    labels_fp = configs.DETECTOR_2D['LABEL_FP']
    one_hot_vec_map = configs.DETECTOR_2D['ONE_HOT_VECTOR_MAP']

    def __init__(self, *args, **kwargs):
        super(Detector2D, self).__init__(*args, **kwargs)
        self._load_labels()

    def inference_verbose(self, data):
        self.img_received = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        self.img_height_received, self.img_width_received, _ = self.img_received.shape
        self.img_resized = cv2.resize(self.img_received, (self.img_resize_size, self.img_resize_size))
        print('[Detector2D]Resizing image from {} to {}'.format(self.img_received.shape, self.img_resized.shape))
        self.img_feed = np.expand_dims(self.img_resized, axis=0)
        self.inference([self.img_feed])
        bboxes_2d, one_hot_vectors = self.post_process()
        print('[Detector2D]boxes 2d are {}\n                one_hot_vectors are {}'.format(bboxes_2d, one_hot_vectors))
        return bboxes_2d, one_hot_vectors

    def _load_labels(self):
        self.label_map = libs.label_map_util.load_labelmap(self.labels_fp)
        self.categories = libs.label_map_util.convert_label_map_to_categories(self.label_map,
                                                                              max_num_classes=self.num_classes,
                                                                              use_display_name=True)
        self.category_index = libs.label_map_util.create_category_index(self.categories)

    def _get_one_hot_vet(self, cls):
        one_hot_vec = np.zeros((3))
        one_hot_vec[self.one_hot_vec_map[cls]] = 1
        print('[Detector2D]Converting {} to {}'.format(cls, one_hot_vec))
        return one_hot_vec

    def post_process(self, threshold=0.2):
        boxes, scores, classes, num_detections = self.prediction
        filtered_results = []
        bb_o = []
        one_hot_vectors = []
        print('[Detector2D]Number of detetcions is {}'.format(num_detections))
        for i in range(0, num_detections):
            score = scores[0][i]
            if score >= threshold:
                print('[Detector2D]Found a detected class with score higher than %s' % score)
                y1, x1, y2, x2 = boxes[0][i]
                y1_o = int(y1 * self.img_height_received)
                x1_o = int(x1 * self.img_width_received)
                y2_o = int(y2 * self.img_height_received)
                x2_o = int(x2 * self.img_width_received)
                predicted_class = self.category_index[classes[0][i]]['name']
                filtered_results.append({
                    "score": score,
                    "bb": boxes[0][i],
                    "bb_o": [x1_o, y1_o, x2_o, y2_o],
                    "img_size": [self.img_height_received, self.img_width_received],
                    "class": predicted_class
                })
                print('[Detector2D]%s: %s, %s' % (predicted_class, score, [x1_o, y1_o, x2_o, y2_o]))
                bb_o.append([x1_o, y1_o, x2_o, y2_o])
                one_hot_vectors.append(self._get_one_hot_vet(predicted_class))
        self._viz(filtered_results)
        return bb_o, one_hot_vectors

    def _viz(self, filtered_results):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        line_type = 2
        offset = 20
        for res in filtered_results:
            x1, y1, x2, y2 = res["bb_o"]
            cv2.rectangle(self.img_received, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(self.img_received, res["class"],
                        (x1 + offset, y1 - offset),
                        font,
                        font_scale,
                        font_color,
                        line_type)
        cv2.imshow('img', self.img_received)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
