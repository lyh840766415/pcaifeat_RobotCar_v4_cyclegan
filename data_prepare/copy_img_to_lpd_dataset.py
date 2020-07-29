import numpy as np
import os
import shutil


INPUT_PATH = "/data/lyh/RobotCar/benchmark_datasets/oxford_day/"
OUTPUT_PATH = "/data/lyh/RobotCar/benchmark_datasets/oxford_lpd_day/"
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
	day_seq = []
	night_seq = []
	all_seqs = sorted(os.listdir(INPUT_PATH))
	night_seq_id = [5,10,13]
	
	'''
	for i,seq in enumerate(all_seqs):
		if i in night_seq_id:
			night_seq.append(seq)
		else:
			day_seq.append(seq)
			
	print(len(day_seq))
	print(len(night_seq))
	exit()
	'''
	
	#for each FOLDER
	for seq_id,seq in enumerate(all_seqs):
		if os.path.isfile(os.path.join(INPUT_PATH,seq)):
			continue
		
		all_pointclouds_1 = []
		all_pointclouds_2 = []
		if os.path.exists(os.path.join(INPUT_PATH,seq,'pointcloud_20m_10overlap')):
			all_pointclouds_1 = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointcloud_20m_10overlap')))
			#mkdir(os.path.join(OUTPUT_PATH,seq,'pointcloud_20m_10overlap'))
		if os.path.exists(os.path.join(INPUT_PATH,seq,'pointcloud_20m')):
			all_pointclouds_2 = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointcloud_20m')))
			#mkdir(os.path.join(OUTPUT_PATH,seq,'pointcloud_20m'))
		
		'''	
		if os.path.exists(os.path.join(INPUT_PATH,seq,CSV_FILENAME_1)):
			print("%s exists"%(os.path.join(INPUT_PATH,seq,CSV_FILENAME_1)))
			shutil.copy(os.path.join(INPUT_PATH,seq,CSV_FILENAME_1),os.path.join(OUTPUT_PATH,seq))
		
		if os.path.exists(os.path.join(INPUT_PATH,seq,CSV_FILENAME_2)):
			print("%s exists"%(os.path.join(INPUT_PATH,seq,CSV_FILENAME_2)))
			shutil.copy(os.path.join(INPUT_PATH,seq,CSV_FILENAME_2),os.path.join(OUTPUT_PATH,seq))
		'''
		print(len(all_pointclouds_1))
		print(len(all_pointclouds_2))
		
		for pointcloud in all_pointclouds_1:
			if not pointcloud.endswith(".png"):
				continue
			
			input_filename = os.path.join(INPUT_PATH,seq,'pointcloud_20m_10overlap',pointcloud)
			
			if seq_id in night_seq_id:
				output_filename = os.path.join(OUTPUT_PATH,seq,'featurecloud_20m_10overlap',"%s.png"%pointcloud[:-4])
			else:
				output_filename = os.path.join(OUTPUT_PATH,seq,'featurecloud_20m_10overlap',"%s.png"%pointcloud[:-4])
				
			print(output_filename)
			shutil.copy(input_filename,output_filename)

		for pointcloud in all_pointclouds_2:
			if not pointcloud.endswith(".png"):
				continue
			
			input_filename = os.path.join(INPUT_PATH,seq,'pointcloud_20m',pointcloud)
			
			if seq_id in night_seq_id:
				output_filename = os.path.join(OUTPUT_PATH,seq,'featurecloud_20m',"%s.png"%pointcloud[:-4])
			else:
				output_filename = os.path.join(OUTPUT_PATH,seq,'featurecloud_20m',"%s.png"%pointcloud[:-4])
			
			
			print(output_filename)
			shutil.copy(input_filename,output_filename)
			
	print("done")


if __name__ == "__main__":
	main()