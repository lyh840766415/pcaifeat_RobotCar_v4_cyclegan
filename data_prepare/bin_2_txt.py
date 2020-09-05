import numpy as np
import os
import shutil


INPUT_PATH = "/data/lyh/benchmark_datasets/oxford/"
OUTPUT_PATH = "/data/lyh/benchmark_datasets/oxford_txt/"
CSV_FILENAME_1 = "pointcloud_locations_20m_10overlap.csv"
CSV_FILENAME_2 = "pointcloud_locations_20m.csv"



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
		
		for pointcloud in all_pointclouds_1:
			pc=np.fromfile(os.path.join(INPUT_PATH,seq,'pointcloud_20m_10overlap',pointcloud), dtype=np.float64)
			if(pc.shape[0]!= 4096*3):
				print("Error in pointcloud shape")
				return np.array([])
				exit()
			pc=np.reshape(pc,(pc.shape[0]//3,3))
			np.savetxt(os.path.join(OUTPUT_PATH,seq,'pointcloud_20m_10overlap',"%s.txt"%(pointcloud[:-4])),pc,fmt="%.3f",delimiter=' ')
		
		for pointcloud in all_pointclouds_2:
			pc=np.fromfile(os.path.join(INPUT_PATH,seq,'pointcloud_20m',pointcloud), dtype=np.float64)
			if(pc.shape[0]!= 4096*3):
				print("Error in pointcloud shape")
				return np.array([])
				exit()
			pc=np.reshape(pc,(pc.shape[0]//3,3))
			np.savetxt(os.path.join(OUTPUT_PATH,seq,'pointcloud_20m',"%s.txt"%(pointcloud[:-4])),pc,fmt="%.3f",delimiter=' ')
	
	print("done")


if __name__ == "__main__":
	main()