import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import random
import re
import cv2
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

INPUT_PATH = "/data/lyh/RobotCar/stereo_centre/"
POS_PATH = "/data/lyh/RobotCar/gps_ins/"
OUTPUT_PATH = "/data/lyh/lab/pcaifeat_RobotCar_v4_cyclegan/data_prepare/day_night/"

x_width=150
y_width=150
p1=[5735712.768124,620084.402381]
p2=[5735611.299219,620540.270327]
p3=[5735237.358209,620543.094379]
p4=[5734749.303802,619932.693364]   
p=[p1,p2,p3,p4]

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

def check_in_test_set(northing, easting, points, x_width, y_width):
	in_test_set=False
	for point in points:
		if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
			in_test_set=True
			break
	return in_test_set

def main():
	
	day_seq = []
	night_seq = []
	night_seq_id = [5,10,13]
	#get the sequence name of day and night
	all_seqs = sorted(os.listdir(INPUT_PATH))
	for i,seq in enumerate(all_seqs):
		if i in night_seq_id:
			night_seq.append(seq)
		else:
			day_seq.append(seq)
			
	#for each day seq, select 10038/42 img, load the image and save to the dataset
	tot_train_saved_img = 0
	tot_test_saved_img = 0
	tot_val_saved_img = 0	
	
	for i,seq in enumerate(night_seq):
		seq_dir = os.path.join(INPUT_PATH,seq)
		timestamp_file = os.path.join(INPUT_PATH,seq,"stereo.timestamps")
		pos_file = os.path.join(POS_PATH,seq,"gps/ins.csv")
		timestamps = np.loadtxt(timestamp_file)
		timestamps[:,1] = 0
		
		pos = pd.read_csv(pos_file,sep=',')
		pos = np.array(pos[['timestamp','northing','easting']])

		pos_tree_data = pos[:,0:2].copy()
		pos_tree_data[:,1] = 0
	
		
		pos_tree = KDTree(pos_tree_data)
		
		dist, ind = pos_tree.query(timestamps, k=1)
		
		all_ind = np.arange(0,ind.shape[0])
		random.shuffle(all_ind)
		cur_seq_saved_img = 0
		cur_ind = 0
		
		while cur_seq_saved_img < 3696:
			cur_pos_ind = ind[all_ind[cur_ind]]
			cur_dist = dist[all_ind[cur_ind]]
			cur_ind = cur_ind + 1
			if cur_dist > 1e5:
				#print("time interval")
				continue
			
			cur_pos = pos[cur_pos_ind,1:3]
			#print(cur_pos[0])
			
			img_path = os.path.join(seq_dir,"stereo/centre","%d.png"%(timestamps[all_ind[cur_ind],0]))
			if(not os.path.exists(img_path)):
				print("img_not exist")
				exit()
			
			loaded_img = load_image_demosaic(img_path)
			
			if(check_in_test_set(cur_pos[0,0], cur_pos[0,1], p, x_width, y_width)):
				save_img_filename = os.path.join(OUTPUT_PATH,"testB","%d_B.png"%(tot_test_saved_img))
				tot_test_saved_img = tot_test_saved_img + 1
				cv2.imwrite(save_img_filename,loaded_img)
				#print("in test set")
				continue;
				
			if(cur_seq_saved_img < 3360):
				save_img_filename = os.path.join(OUTPUT_PATH,"trainB","%d_B.png"%(tot_train_saved_img))
				tot_train_saved_img = tot_train_saved_img + 1
			else:
				save_img_filename = os.path.join(OUTPUT_PATH,"valB","%d_B.png"%(tot_val_saved_img))
				tot_val_saved_img = tot_val_saved_img + 1
			cv2.imwrite(save_img_filename,loaded_img)
			
			#print(img_path)
			cur_seq_saved_img = cur_seq_saved_img + 1
			
		
		print(cur_seq_saved_img)
		print(cur_ind)
		print(seq,"finish")	

	tot_train_saved_img = 0
	tot_test_saved_img = 0
	tot_val_saved_img = 0	
	for i,seq in enumerate(day_seq):
		seq_dir = os.path.join(INPUT_PATH,seq)
		timestamp_file = os.path.join(INPUT_PATH,seq,"stereo.timestamps")
		pos_file = os.path.join(POS_PATH,seq,"gps/ins.csv")
		timestamps = np.loadtxt(timestamp_file)
		timestamps[:,1] = 0
		
		pos = pd.read_csv(pos_file,sep=',')
		pos = np.array(pos[['timestamp','northing','easting']])

		pos_tree_data = pos[:,0:2].copy()
		pos_tree_data[:,1] = 0
	
		
		pos_tree = KDTree(pos_tree_data)
		
		dist, ind = pos_tree.query(timestamps, k=1)
		
		all_ind = np.arange(0,ind.shape[0])
		random.shuffle(all_ind)
		cur_seq_saved_img = 0
		cur_ind = 0
		
		while cur_seq_saved_img < 264:
			cur_pos_ind = ind[all_ind[cur_ind]]
			cur_dist = dist[all_ind[cur_ind]]
			cur_ind = cur_ind + 1
			if cur_dist > 1e5:
				#print("time interval")
				continue
			
			cur_pos = pos[cur_pos_ind,1:3]
			#print(cur_pos[0])
			
			img_path = os.path.join(seq_dir,"stereo/centre","%d.png"%(timestamps[all_ind[cur_ind],0]))
			if(not os.path.exists(img_path)):
				print("img_not exist")
				exit()
			
			loaded_img = load_image_demosaic(img_path)
			
			if(check_in_test_set(cur_pos[0,0], cur_pos[0,1], p, x_width, y_width)):
				save_img_filename = os.path.join(OUTPUT_PATH,"testA","%d_A.png"%(tot_test_saved_img))
				tot_test_saved_img = tot_test_saved_img + 1
				cv2.imwrite(save_img_filename,loaded_img)
				#print("in test set")
				continue;
				
			if(cur_seq_saved_img < 240):
				save_img_filename = os.path.join(OUTPUT_PATH,"trainA","%d_A.png"%(tot_train_saved_img))
				tot_train_saved_img = tot_train_saved_img + 1
			else:
				save_img_filename = os.path.join(OUTPUT_PATH,"valA","%d_A.png"%(tot_val_saved_img))
				tot_val_saved_img = tot_val_saved_img + 1
			cv2.imwrite(save_img_filename,loaded_img)
			
			#print(img_path)
			cur_seq_saved_img = cur_seq_saved_img + 1
			
		
		print(cur_seq_saved_img)
		print(cur_ind)
		print(seq,"finish")	

		
	
	#for each night seq, select 10038/3 img, load the image and save to the dataset
		

if __name__ == "__main__":
	main()