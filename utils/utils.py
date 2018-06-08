import numpy as np
from configs import configs


def viz(pc, centers, corners_3d, pc_origin):
	import mayavi.mlab as mlab
	fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
	                  fgcolor=None, engine=None, size=(500, 500))
	mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], mode='sphere',
	              colormap='gnuplot', scale_factor=0.1, figure=fig)
	mlab.points3d(centers[:, 0], centers[:, 1], centers[:, 2], mode='sphere',
	              color=(1, 0, 1), scale_factor=0.3, figure=fig)
	mlab.points3d(corners_3d[:, 0], corners_3d[:, 1], corners_3d[:, 2], mode='sphere',
	              color=(1, 1, 0), scale_factor=0.3, figure=fig)
	mlab.points3d(pc_origin[:, 0], pc_origin[:, 1], pc_origin[:, 2], mode='sphere',
	              color=(0, 1, 0), scale_factor=0.05, figure=fig)
	'''
        Green points are original PC from KITTI
        White points are PC feed into the network
        Red point is the predicted center
        Yellow point the post-processed predicted bounding box corners
    '''
	raw_input("Press any key to continue")

def viz_single(pc):
	import mayavi.mlab as mlab
	fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4),
	                  fgcolor=None, engine=None, size=(500, 500))
	mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], mode='sphere',
	              colormap='gnuplot', scale_factor=0.1, figure=fig)
	'''
        Green points are original PC from KITTI
        White points are PC feed into the network
        Red point is the predicted center
        Yellow point the post-processed predicted bounding box corners
    '''
	raw_input("Press any key to continue")


def load_velo_scan(velo_filename):
	scan = np.fromfile(velo_filename, dtype=np.float32)
	scan = scan.reshape((-1, 4))
	return scan


def read_calib_file(filepath):
	''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
	data = {}
	with open(filepath, 'r') as f:
		for line in f.readlines():
			line = line.rstrip()
			if len(line) == 0: continue
			key, value = line.split(':', 1)
			# The only non-float values in these files are dates, which
			# we don't care about anyway
			try:
				data[key] = np.array([float(x) for x in value.split()])
			except ValueError:
				pass

	return data


def class2size(pred_cls, residual):
	''' Inverse function to size2class. '''
	mean_size = configs.g_type_mean_size[configs.g_class2type[pred_cls]]
	return mean_size + residual


def class2angle(pred_cls, residual, num_class, to_label_format=True):
	''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
	angle_per_class = 2 * np.pi / float(num_class)
	angle_center = pred_cls * angle_per_class
	angle = angle_center + residual
	if to_label_format and angle > np.pi:
		angle = angle - 2 * np.pi
	return angle


def get_3d_box(box_size, heading_angle, center):
	''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

	def roty(t):
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c, 0, s],
		                 [0, 1, 0],
		                 [-s, 0, c]])

	R = roty(heading_angle)
	l, w, h = box_size
	x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
	z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
	corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
	corners_3d[0, :] = corners_3d[0, :] + center[0]
	corners_3d[1, :] = corners_3d[1, :] + center[1]
	corners_3d[2, :] = corners_3d[2, :] + center[2]
	corners_3d = np.transpose(corners_3d)
	return corners_3d
