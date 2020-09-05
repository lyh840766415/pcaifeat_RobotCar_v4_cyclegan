import numpy as np
import os
import shutil


INPUT_PATH = "/data/lyh/RobotCar/pc_img_ground_0310/20m_20dis/"
OUTPUT_PATH = "/data/lyh/RobotCar/pc_img_ground_0310/20m_20dis_color_resize/"
CSV_FILENAME = "pointcloud_locations_20m_20dis.csv"


def mkdir(path):
	if os.path.exists(path):
		print("%s is already exist!"%(path))
	else:
		os.makedirs(path)	
		print("%s is created"%(path))
		
	
		

def main():
	#get all folder from the INPUT_PATH
	all_seqs = sorted(os.listdir(INPUT_PATH))
	
	#for each FOLDER
	for seq in all_seqs:
		if os.path.isfile(os.path.join(INPUT_PATH,seq)):
			continue
		
		#print(seq)
		mkdir(os.path.join(OUTPUT_PATH,seq,'pointclouds'))
		
		if not os.path.exists(os.path.join(INPUT_PATH,seq,CSV_FILENAME)):
			print("error, %s do not exists"%(os.path.join(INPUT_PATH,seq,CSV_FILENAME)))
			exit()
			
		shutil.copy(os.path.join(INPUT_PATH,seq,CSV_FILENAME),os.path.join(OUTPUT_PATH,seq))
		
		all_pointclouds = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointclouds')))
		
		for pointcloud in all_pointclouds:
			if pointcloud[-10:] == 'imgpos.txt':
				shutil.copy(os.path.join(INPUT_PATH,seq,'pointclouds',pointcloud),os.path.join(OUTPUT_PATH,seq,'pointclouds'))
			if pointcloud[-4:] == '.bin':
				shutil.copy(os.path.join(INPUT_PATH,seq,'pointclouds',pointcloud),os.path.join(OUTPUT_PATH,seq,'pointclouds'))
			
			
	print("done")


if __name__ == "__main__":
	main()