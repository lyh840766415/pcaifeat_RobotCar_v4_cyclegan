import pickle
import numpy as np
import os
import cv2
import random
import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

BASE_PATH = "/"

def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Trajectories Loaded.")
		return trajectories

def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries

def get_pc_img_match_dict(filename):
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("point image match Loaded.")
		return queries	

def load_pc_file_32(filename):
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((4096,3))
		exit()
		
	#returns Nx3 matrix
	#print(filename)
	pc=np.fromfile(filename, dtype=np.float32)
	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])
		print(filename)
		exit()

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc

def load_pc_file_lpd(filename):
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((4096,13))
		exit()
		
	#returns Nx13 matrix (3 pose 10 handcraft features)
	pc=np.fromfile(filename, dtype=np.float64)

	if(pc.shape[0]!= 4096*13):
		print("Error in pointcloud shape")
		print(pc.shape)
		print(filename)
		#return np.array([])
		return np.zeros([4096,13])

	pc=np.reshape(pc,(pc.shape[0]//13,13))

	# preprocessing data
	# Normalization
	pc[:,3:12] = ((pc-pc.min(axis=0))/(pc.max(axis=0)-pc.min(axis=0)))[:,3:12]
	pc[np.isnan(pc)] = 0.0
	pc[np.isinf(pc)] = 1.0

	return pc

def load_kdtree(filename):
	if not os.path.exists(filename):
		print(filename)
		exit()
	
	with open(filename, 'rb') as f:
		kdtree = pickle.load(f)
	return kdtree
	
		
def load_pc_file_64(filename):
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((4096,3))
		exit()
		
	#returns Nx3 matrix
	#print(filename)
	pc=np.fromfile(filename, dtype=np.float64)
	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])
		print(filename)
		exit()

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	pc = pc.astype(np.float32)
	return pc

def load_pc_files(filenames):
	pcs=[]
	for filename in filenames:
		#print(filename)
		pc,success=load_pc_file(filename)
		if not success:
			return np.array([]),False
		#if(pc.shape[0]!=4096):
		#	continue
		pcs.append(pc)
	pcs=np.array(pcs)
	return pcs,True

def load_pc_file_save_2_txt(filename,output_dir):
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((4096,3)),True
		
	#returns Nx3 matrix
	#print(filename)
	pc=np.fromfile(filename, dtype=np.float64)
	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([]),True

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	np.savetxt(os.path.join(output_dir,"%s.xyz"%(filename[-20:-4])), pc, fmt="%.5f", delimiter = ',')
	return pc,True
	
def load_image(filename):
	#return scaled image
	if not os.path.exists(filename):
		print(filename)
		return np.zeros((240,320,3))
	
	img = cv2.imread(filename)
	img = cv2.resize(img,(320,240))
	
	return img

def load_image_demosaic(filename):
	BAYER_STEREO = 'gbrg'
	BAYER_MONO = 'rggb'
	camera = re.search('(stereo|mono_(left|right|rear))', filename).group(0)
	if camera == 'stereo':
		pattern = BAYER_STEREO
	else:
		pattern = BAYER_MONO
	
	img = Image.open(filename)
	print(img)
	img = demosaic(img, pattern)
	
	#img = cv2.imread(filename)
	
	print(img.shape)
	
	#img = cv2.resize(img,(256,256))
	#img = demosaic(img, pattern)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	
	return np.array(img).astype(np.uint8)
		

def load_images(filenames):
	imgs=[]
	for filename in filenames:
		#print(filename)
		img,success=load_image(filename)
		if not success:
			return np.array([]),False
		imgs.append(img)
	imgs=np.array(imgs)
	return imgs,True

def load_img_pc_lpd(load_pc_filenames,load_img_filenames,pool):
	pcs = []
	imgs = []
	if load_pc_filenames != None:
		pcs = pool.map(load_pc_file_lpd,load_pc_filenames)
		pcs = np.array(pcs)
	if load_img_filenames != None:
		imgs = pool.map(load_image,load_img_filenames)
		imgs=np.array(imgs)

	return pcs,imgs

def load_img_pc(load_pc_filenames,load_img_filenames,pool,pc_64bit):
	pcs = []
	imgs = []
	if load_pc_filenames != None:
		if pc_64bit:
			pcs = pool.map(load_pc_file_64,load_pc_filenames)
		else:
			pcs = pool.map(load_pc_file_32,load_pc_filenames)
		pcs = np.array(pcs)
	if load_img_filenames != None:
		imgs = pool.map(load_image,load_img_filenames)
		imgs=np.array(imgs)

	return pcs,imgs

def load_img_pc_kdtree(load_pc_filenames,load_kdtree_filenames,load_img_filenames,pool,pc_64bit):
	pcs = []
	imgs = []
	if load_pc_filenames != None:
		if pc_64bit:
			pcs = pool.map(load_pc_file_64,load_pc_filenames)
		else:
			pcs = pool.map(load_pc_file_32,load_pc_filenames)
		pcs = np.array(pcs)
	if load_img_filenames != None:
		imgs = pool.map(load_image,load_img_filenames)
		imgs=np.array(imgs)
	if load_kdtree_filenames != None:
		kdtrees = pool.map(load_kdtree,load_kdtree_filenames)
		kdtrees = np.array(kdtrees)

	return pcs,kdtrees,imgs

def load_img_pc_from_net(load_pc_filenames,load_img_filenames,pool,pc_64bit):
	pcs = []
	imgs = []
	
	NET_PATH = "/media/lyh/shared_space/lyh/dataset/ROBOTCAR/"
	if load_pc_filenames != None:
		for i in range(len(load_pc_filenames)):
			substr = load_pc_filenames[i].split('/')
			substr = [x for x in substr if x != '']
			load_pc_filenames[i] = os.path.join(NET_PATH,substr[-5],substr[-4],substr[-3],substr[-2],substr[-1])
		if pc_64bit:
			pcs = pool.map(load_pc_file_64,load_pc_filenames)
		else:
			pcs = pool.map(load_pc_file_32,load_pc_filenames)
		pcs = np.array(pcs)
		
	if load_img_filenames != None:
		for i in range(len(load_img_filenames)):
			substr = load_img_filenames[i].split('/')
			substr = [x for x in substr if x != '']
			load_img_filenames[i] = os.path.join(NET_PATH,substr[-5],substr[-4],substr[-3],substr[-2],substr[-1])

		imgs = pool.map(load_image,load_img_filenames)
		imgs=np.array(imgs)
	
	return pcs,imgs
	
def load_img_pc_from_net_demosaic(load_pc_filenames,load_img_filenames,pool):
	pcs = []
	imgs = []
	if load_pc_filenames != None:
		pcs = pool.map(load_pc_file,load_pc_filenames)
		pcs = np.array(pcs)
	if load_img_filenames != None:
		NET_PATH = "/media/lyh/shared_space/lyh/dataset/ROBOTCAR/"
		for i in range(len(load_img_filenames)):
			substr = load_img_filenames[i].split('/')
			substr = [x for x in substr if x != '']		
			load_img_filenames[i] = os.path.join(NET_PATH,substr[-5],substr[-4],substr[-3],substr[-2],substr[-1])
		imgs = pool.map(load_image_demosaic,load_img_filenames)
		imgs=np.array(imgs)
	
	return pcs,imgs