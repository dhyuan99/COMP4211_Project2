from PIL import Image
import torch
from Encoder import *
from Decoder import *
import pandas as pd
from utils import *
import argparse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

E = nn.Sequential(resnet34(), nn.Linear(1000,99))
D = invresnet34(input_dims=99)
E.load_state_dict(torch.load('../model/E.pth'))
D.load_state_dict(torch.load('../model/D.pth'))
E.eval()
D.eval()

parser = argparse.ArgumentParser(description='Setting for Question 8')
parser.add_argument('-example', '--example', help='comparison or cluster or tsne or transfer.')
args = parser.parse_args()

if args.example == 'comparison':
	img_list = ['me_2013', 'me_2017', 'father_2013', 'mother_2013']
	dist = pd.DataFrame(np.zeros([4,4]), columns=img_list, index=img_list)
	for img1 in img_list:
		for img2 in img_list:
			img1_path = '../examples/comparison/'+img1+'.jpg'
			img2_path = '../examples/comparison/'+img2+'.jpg'
			dist[img1][img2] = diff(img1_path, img2_path, E).item()
	print(dist)

if args.example == 'cluster':
	imgs_name, imgs_list = get_imgs_list('../examples/cluster/')
	kmean_result = cluster(imgs_name, imgs_list, E, KMeans, k=2)
	print('The result given by kmean clustering:')
	print(kmean_result)
	print('\n')

	agg_result = cluster(imgs_name, imgs_list, E, AgglomerativeClustering, k=2)
	print('The result given by Agglomerative clustering:\n')
	print(agg_result)
	print('\n')

if args.example == 'tsne':
	imgs_name, imgs_list = get_imgs_list('../examples/cluster/')
	emb = retrieve_emb(imgs_list, E)
	emb_2d = TSNE(n_components=2).fit_transform(emb)
	fig, ax = plt.subplots()
	for i in range(len(emb_2d)):
		if 's' in imgs_name[i]:
			ax.scatter(emb_2d[i,0], emb_2d[i,1], color='r')
		if 'z' in imgs_name[i]:
			ax.scatter(emb_2d[i,0], emb_2d[i,1], color='b')
		ax.annotate(imgs_name[i].strip('.jpg'), (emb_2d[i,0], emb_2d[i,1]))
	fig.savefig('../examples/tsne/tsne.jpg')
	print(imgs_name)

if args.example == 'transfer':
	img1 = Image.open('../examples/transfer/me_2013.jpg')
	img2 = Image.open('../examples/transfer/me_2017.jpg')
	img1 = img1.resize((256,256))
	img2 = img2.resize((256,256))
	img1 = torch.Tensor(np.asarray(img1)).reshape([1,3,256,256])
	img2 = torch.Tensor(np.asarray(img2)).reshape([1,3,256,256])
	z1 = E(img1)
	z2 = E(img2)
	v = z2 - z1
	for t in range(0,21):
		z = z1 + v * t / 20
		xhat = D(z)
		plt.figure()
		plt.imshow(xhat[0].permute([1,2,0]).detach().numpy())
		plt.gca().set_axis_off()
		plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
		plt.margins(0,0)
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.savefig('../examples/transfer/results/'+str(t/20)+'.jpg')
		plt.close()	
		

