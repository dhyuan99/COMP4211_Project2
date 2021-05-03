import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os

def diff(img1_path, img2_path, E):
	img1 = Image.open(img1_path)
	img2 = Image.open(img2_path)
	img1 = img1.resize((256,256))
	img2 = img2.resize((256,256))
	img1 = torch.Tensor(np.asarray(img1)).permute([2,0,1]).reshape([1,3,256,256])
	img2 = torch.Tensor(np.asarray(img2)).permute([2,0,1]).reshape([1,3,256,256])
	return torch.sqrt(F.mse_loss(E(img1), E(img2)))

def cluster(imgs_name, imgs_list, E, alg, k=2):
	# imgs_list is a list of 3*256*256 Tensors.
	emb = retrieve_emb(imgs_list, E)
	kmeans = alg(n_clusters=k).fit(emb)
	cluster_result = pd.DataFrame(kmeans.labels_, columns=['label'], index=imgs_name)
	cluster_result = cluster_result.sort_values(by='label')
	return cluster_result


def retrieve_emb(imgs_list, E):
	emb = np.zeros([len(imgs_list), 99])
	for i in range(len(imgs_list)):
		img = imgs_list[i]
		img = img.reshape([1,3,256,256])
		emb[i] = E(img)[0].detach().numpy()
	return emb

def get_imgs_list(img_path):
	Dir = os.listdir(img_path)
	imgs_list = []
	for d in Dir:
		img = Image.open(img_path+d)
		img = img.resize((256,256))
		img = torch.Tensor(np.asarray(img)).permute([2,0,1])
		imgs_list.append(img)
	return Dir, imgs_list
