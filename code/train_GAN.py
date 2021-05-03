import torch
from Encoder import *
from Decoder import *
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt

z_dim = 99
lr = 1e-3
batch_size = 64
eps = 1e-6

D = invresnet34(input_dims=z_dim).cuda()
Dx = nn.Sequential(resnet18(), nn.Linear(1000,1), nn.Sigmoid()).cuda()

D_solver = optim.Adam(D.parameters(), lr=lr)
Dx_solver = optim.Adam(Dx.parameters(), lr=lr)

transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

trainset = torchvision.datasets.ImageFolder("../data/train", transform=transform, target_transform=None)
data_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=2,drop_last=True)

for it in range(10000):
	for batch_idx, batch_item in enumerate(data_loader):
		for _ in range(3):
			X = torch.randn([len(batch_item[0]), z_dim]).cuda()
			Y = Variable(batch_item[0]).cuda()
			Yhat = D(X)
			Y_fake = Dx(Yhat)

			loss = -torch.mean(torch.log(eps+Y_fake))
			loss.backward()
			D_solver.step()
			D.zero_grad()
			Dx.zero_grad()

		for _ in range(1):
			X = torch.randn([len(batch_item[0]), z_dim]).cuda()
			Y = Variable(batch_item[0]).cuda()
			Yhat = D(X)
			Y_real = Dx(Y)
			Y_fake = Dx(Yhat)

			loss = -torch.mean(torch.log(Y_real+eps)+torch.log(1+eps-Y_fake))
			loss.backward()
			Dx_solver.step()
			D.zero_grad()
			Dx.zero_grad()

		if batch_idx % 100 == 0:
			print(loss.item())

			plt.figure()
			plt.subplot(1,2,1)
			plt.imshow(Y[0].permute([1,2,0]).cpu().detach().numpy())
			plt.subplot(1,2,2)
			plt.imshow(Yhat[0].permute([1,2,0]).cpu().detach().numpy())
			plt.savefig('../result/'+str(it)+'.jpg')
			plt.close()

			torch.save(D.state_dict(), '../model/D.pth')