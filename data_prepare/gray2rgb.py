import numpy as np
import os
import shutil
import re
import cv2
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(40)


INPUT_PATH = "/data/lyh/RobotCar/pc_img_ground_0310/20m_10dis/"
#OUTPUT_PATH = "/data/lyh/RobotCar/pc_img_ground_0310/20m_20dis_color/"
OUTPUT_PATH_2 = "/data/lyh/RobotCar/pc_img_ground_0310/20m_10dis_color_resize/"
CSV_FILENAME = "pointcloud_locations_20m_10dis.csv"


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
	#cv2.imwrite(os.path.join(OUTPUT_PATH,substr[-3],substr[-2],substr[-1]),img)
	img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
	cv2.imwrite(os.path.join(OUTPUT_PATH_2,substr[-3],substr[-2],substr[-1]),img)


def main():
	#get all folder from the INPUT_PATH
	all_seqs = sorted(os.listdir(INPUT_PATH))
	
	#for each FOLDER
	for seq in all_seqs:
		if os.path.isfile(os.path.join(INPUT_PATH,seq)):
			continue
		
		#print(seq)
		#mkdir(os.path.join(OUTPUT_PATH,seq,'pointclouds'))
		mkdir(os.path.join(OUTPUT_PATH_2,seq,'pointclouds'))
		#if not os.path.exists(os.path.join(INPUT_PATH,seq,CSV_FILENAME)):
		#	print("error, %s do not exists"%(os.path.join(INPUT_PATH,seq,CSV_FILENAME)))
		#	exit()
			
		#shutil.copy(os.path.join(INPUT_PATH,seq,CSV_FILENAME),os.path.join(OUTPUT_PATH,seq))
		
		allfile  = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointclouds')))
		
		images = []
		for cur_file in allfile:
			if cur_file[-17:] == 'stereo_centre.png':
				#print(os.path.join(INPUT_PATH,seq,'pointclouds',cur_file))
				images.append(os.path.join(INPUT_PATH,seq,'pointclouds',cur_file))
		
		imgs = pool.map(load_image_demosaic,images)
	
	print("done")


if __name__ == "__main__":
	main()