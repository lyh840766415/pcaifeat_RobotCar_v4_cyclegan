import numpy as np
import os
import shutil


INPUT_PATH = "/data/lyh/RobotCar/benchmark_datasets/"
OUTPUT_PATH_A = "/data/lyh/lab/pytorch-CycleGAN-and-pix2pix/datasets/day_night_test/testA/"
OUTPUT_PATH_B = "/data/lyh/lab/pytorch-CycleGAN-and-pix2pix/datasets/day_night_test/testB/"
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
		if os.path.exists(os.path.join(INPUT_PATH,seq,'pointcloud_20m')):
			all_pointclouds_2 = sorted(os.listdir(os.path.join(INPUT_PATH,seq,'pointcloud_20m')))
		
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
				output_filename = os.path.join(OUTPUT_PATH_B,seq+'+pointcloud_20m_10overlap+'+pointcloud)
			else:
				output_filename = os.path.join(OUTPUT_PATH_A,seq+'+pointcloud_20m_10overlap+'+pointcloud)
			print(output_filename)
			shutil.copy(input_filename,output_filename)

		for pointcloud in all_pointclouds_2:
			if not pointcloud.endswith(".png"):
				continue
			
			input_filename = os.path.join(INPUT_PATH,seq,'pointcloud_20m',pointcloud)
			if seq_id in night_seq_id:
				output_filename = os.path.join(OUTPUT_PATH_B,seq+'+pointcloud_20m+'+pointcloud)
			else:
				output_filename = os.path.join(OUTPUT_PATH_A,seq+'+pointcloud_20m+'+pointcloud)
			
			print(output_filename)
			shutil.copy(input_filename,output_filename)
			
	print("done")


if __name__ == "__main__":
	main()