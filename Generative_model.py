import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self, input_shape):
		super(Net, self).__init__()

		self.encoder = nn.Sequential(
									nn.Linear(in_features=input_shape, out_features=256),
									nn.ReLU(),
									nn.Linear(in_features=256, out_features=128),
									nn.ReLU(),
									nn.Linear(in_features=128, out_features=32)),
									nn.ReLU(),
		self.decoder = nn.Sequential(
									nn.Linear(in_features=32, out_features=128),
									nn.ReLU(),
									nn.Linear(in_features=128, out_features=256),
									nn.ReLU(),
									nn.Linear(in_features=256, out_features=input_shape))

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)

		return x