import numpy as np
import os
import shutil


INPUT_PATH = "/data/lyh/RobotCar/tmp/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left/"
OUTPUT_PATH = "/data/lyh/RobotCar/tmp/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left_txt/"



def mkdir(path):
	if os.path.exists(path):
		print("%s is already exist!"%(path))
	else:
		os.makedirs(path)	
		print("%s is created"%(path))
		
	
		

def main():
	#get all folder from the INPUT_PATH
	all_pointclouds = sorted(os.listdir(INPUT_PATH))
	
	print(len(all_pointclouds))
	if not os.path.exists(OUTPUT_PATH):
		mkdir(OUTPUT_PATH)
		
	for pointcloud in all_pointclouds:
		if pointcloud[-4:] != '.bin':
			continue
			
		pc = np.fromfile(os.path.join(INPUT_PATH, pointcloud), dtype=np.float32)
		pc = np.reshape(pc,(4,-1))
		pc = pc.T		
		print(pc.shape)
		
		'''
		if(pc.shape[0]!= 4096*3):
			print("Error in pointcloud shape")
			return np.array([])
			exit()
		'''
		
		print(os.path.join(OUTPUT_PATH,"%s.txt"%(pointcloud[:-4])))
		
		np.savetxt(os.path.join(OUTPUT_PATH,"%s.txt"%(pointcloud[:-4])),pc,fmt="%.3f",delimiter=' ')
		
	print("done")


if __name__ == "__main__":
	main()