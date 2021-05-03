import torch
from Encoder import *
from Decoder import *
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
z_dim = 99

D = invresnet34(input_dims=z_dim).cuda()
D.load_state_dict(torch.load('../model/D.pth'))
D.eval()
for param in D.parameters():
	param.requires_grad = False

E = nn.Sequential(resnet34(), nn.Linear(1000,z_dim)).cuda()
E_solver = optim.Adam(E.parameters(), lr=1e-3)

for _ in range(10000):
	z = torch.randn([batch_size, z_dim]).cuda()
	X = D(z)
	zhat = E(X)
	loss = F.mse_loss(z, zhat)
	loss.backward()
	E_solver.step()
	E.zero_grad()

	print(loss.item())
	torch.save(E.state_dict(), '../model/E.pth')