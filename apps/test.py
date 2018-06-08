import cv2
from models import server
from utils import utils
from configs import configs

# Reading example image
img = cv2.imread('{}'.format(configs.TEST_DATA_FP['img']))

# Reading example points cloud
pclds = utils.load_velo_scan('{}'.format(configs.TEST_DATA_FP['pclds']))

test_input = {'img': img, 'pclds': pclds}
server_ins = server.Server()
server_ins.predict(test_input)
