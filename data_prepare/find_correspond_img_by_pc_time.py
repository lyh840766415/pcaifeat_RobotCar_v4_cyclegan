import numpy as np
import os
import re
import shutil
import cv2
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from sklearn.neighbors import KDTree

IMG_PATH = "/data/lyh/RobotCar/stereo_centre/"
INPUT_PATH = "/data/lyh/benchmark_datasets/oxford/"
OUTPUT_PATH = "/data/lyh/benchmark_datasets/oxford_img/"
CSV_FILENAME_1 = "pointcloud_locations_20m_10overlap.csv"
CSV_FILENAME_2 = "pointcloud_locations_20m.csv"



def mkdir(path):
	if os.path.exists(path):
		print("%s is already exist!"%(path))
	else:
		os.makedirs(path)	
		print("%s is created"%(path))

def load_image_demosaic(filename):
	BAYER_STEREO = 'gbrg'
	BAYER_MONO = 'rggb'
	camera = re.search('(stereo|mono_(left|right|rear))', filename).group(0)
	if camera == 'stereo':
		pattern = BAYER_STEREO
	else:
		pattern = BAYER_MONO
	
	img = Image.open(filename)
	img = demosaic(img, pattern)
	img = np.array(img).astype(np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	substr = filename.split('/')
	substr = [x for x in substr if x != '']		
	img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
	return img
		

def main():
	#get all folder from the INPUT_PATH
	all_seqs = sorted(os.listdir(INPUT_PATH))
	
	#for each FOLDER
	for seq in all_seqs:
		#load the timestamps
		timestamps = np.loadtxt(os.path.join(IMG_PATH,seq,'stereo.timestamps'))
		timestamps[:,1] = 0
		timestamps_tree = KDTree(timestamps)
		print(timestamps_tree)
		
		
		if os.path.isfile(os.path.join(INPUT_PATH,seq)):
			continue
		
		all_pointclouds_1 = []
		all_pointclouds_2 = []
		if os.path.exists(os.path.join(INPUT_PATH,seq,'pointcloud_20m_10overlap')):
			all_pointclouds_1 = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointcloud_20m_10overlap')))
			mkdir(os.path.join(OUTPUT_PATH,seq,'pointcloud_20m_10overlap'))
		if os.path.exists(os.path.join(INPUT_PATH,seq,'pointcloud_20m')):
			all_pointclouds_2 = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointcloud_20m')))
			mkdir(os.path.join(OUTPUT_PATH,seq,'pointcloud_20m'))
		
		if os.path.exists(os.path.join(INPUT_PATH,seq,CSV_FILENAME_1)):
			print("%s exists"%(os.path.join(INPUT_PATH,seq,CSV_FILENAME_1)))
			shutil.copy(os.path.join(INPUT_PATH,seq,CSV_FILENAME_1),os.path.join(OUTPUT_PATH,seq))
		
		if os.path.exists(os.path.join(INPUT_PATH,seq,CSV_FILENAME_2)):
			print("%s exists"%(os.path.join(INPUT_PATH,seq,CSV_FILENAME_2)))
			shutil.copy(os.path.join(INPUT_PATH,seq,CSV_FILENAME_2),os.path.join(OUTPUT_PATH,seq))
		
		print(len(all_pointclouds_1))
		print(len(all_pointclouds_2))
		
		all_pointclouds_1_ts = [int(x[:-4]) for x in all_pointclouds_1]
		all_pointclouds_2_ts = [int(x[:-4]) for x in all_pointclouds_2]
		all_pointclouds_1_ts = np.array(all_pointclouds_1_ts)
		all_pointclouds_2_ts = np.array(all_pointclouds_2_ts)
		
		all_pointclouds_1_ts = np.expand_dims(all_pointclouds_1_ts,1)
		all_pointclouds_2_ts = np.expand_dims(all_pointclouds_2_ts,1)
		all_pointclouds_1_ts = np.tile(all_pointclouds_1_ts,[1,2])
		all_pointclouds_2_ts = np.tile(all_pointclouds_2_ts,[1,2])
		all_pointclouds_1_ts[:,1] = 0
		all_pointclouds_2_ts[:,1] = 0
		print(all_pointclouds_1_ts.shape)
		print(all_pointclouds_2_ts.shape)
		
		if len(all_pointclouds_1) > 0:
			dist_1, ind_1 = timestamps_tree.query(all_pointclouds_1_ts, k=1)
		if len(all_pointclouds_2) > 0:
			dist_2, ind_2 = timestamps_tree.query(all_pointclouds_2_ts, k=1)
		
		
		for i,pointcloud in enumerate(all_pointclouds_1):
			#determine fake neighbour
			if dist_1[i] > 1e5:
				print("fake neighbour")
				exit(-1)
			
			#determine whether the stereo_img_exist
			correspond_img_filename = os.path.join(IMG_PATH,seq,"stereo/centre","%d.png"%(timestamps[ind_1[i],0]))
			if not os.path.exists(correspond_img_filename):
				print("file does not exists")
				exit(-2)	
			#load the img, give color and resize
			correspond_img = load_image_demosaic(correspond_img_filename)
			save_img_filename = os.path.join(OUTPUT_PATH,seq,'pointcloud_20m_10overlap',"%s_stereo_centre.png"%(pointcloud[:-4]))
			cv2.imwrite(save_img_filename,correspond_img)

		
		for i,pointcloud in enumerate(all_pointclouds_2):
			#determine fake neighbour
			if dist_2[i] > 1e5:
				print("fake neighbour")
				exit(-1)

			#determine whether the stereo_img_exist
			correspond_img_filename = os.path.join(IMG_PATH,seq,"stereo/centre","%d.png"%(timestamps[ind_2[i],0]))
			if not os.path.exists(correspond_img_filename):
				print("file does not exists")
				exit(-2)	
			#load the img, give color and resize
			correspond_img = load_image_demosaic(correspond_img_filename)
			save_img_filename = os.path.join(OUTPUT_PATH,seq,'pointcloud_20m',"%s_stereo_centre.png"%(pointcloud[:-4]))
			cv2.imwrite(save_img_filename,correspond_img)
		
		print(seq," finish")	
		
	
	print("done")


if __name__ == "__main__":
	main()