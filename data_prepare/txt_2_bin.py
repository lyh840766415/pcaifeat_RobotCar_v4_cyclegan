import numpy as np
import os
import shutil


INPUT_PATH = "/data/lyh/lab/pointnetvlad/submap_generation/output_with_img/20m_10dis/"
OUTPUT_PATH = "/data/lyh/RobotCar/pc_img_without_ground_0320/20m_10dis_color_resize/"
CSV_FILENAME = "pointcloud_locations_20m_10dis.csv"


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
		imgpos_cnt = 0
		pc_cnt = 0
		for pointcloud in all_pointclouds:
			if pointcloud[-4:] != '.txt':
				continue
			if len(pointcloud) == 20:
				pc_cnt += 1
			else:
				imgpos_cnt += 1
		
		if pc_cnt != imgpos_cnt:
			print("pc num = ",pc_cnt)
			print("img num = ",imgpos_cnt)

		
		for pointcloud in all_pointclouds:
			if pointcloud[-4:] != '.txt':
				continue
			
			if len(pointcloud) == 27:
				shutil.copy(os.path.join(INPUT_PATH,seq,'pointclouds',pointcloud),os.path.join(OUTPUT_PATH,seq,'pointclouds'))
				continue				
			if len(pointcloud) != 20:
				print("filename length = ",len(pointcloud))
				continue
			pc=np.loadtxt(os.path.join(INPUT_PATH,seq,'pointclouds',pointcloud),dtype=np.float64)
			
			pc.tofile(os.path.join(OUTPUT_PATH,seq,'pointclouds',"%s.bin"%(pointcloud[:-4])))
	
	print("done")


if __name__ == "__main__":
	main()