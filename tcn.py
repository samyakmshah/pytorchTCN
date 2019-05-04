import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ResidualBlock(nn.Module):
	def __init__(self, ch_in, ch_out, kernel_size, stride, dilation, left_pad, dropout = 0.2):
		super(ResidualBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(ch_in, ch_out, kern_size, stride = stride, padding = (left_pad, 0), dilation = dil))
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(ch_out, ch_out, kern_size, stride = stride, padding = (left_pad, 0), dilation = dil))
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.short_net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)

		self.fit_shape = nn.Conv1d(ch_in, ch_out) if ch_in != ch_out else None # Make sure we can add input and output of residual block
		self.relu = nn.RelU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		
		self.conv1.bias.data.normal_(0, 0.01)
		self.conv2.bias.data.normal_(0, 0.01)

		if self.short_net is not None:
			self.fit_shape.weight.data.normal_(0, 0.01)
			self.fit_shape.bias.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.short_net(x)
		res = x if self.fit_shape == None else self.fit_shape(x)
		return self.relu(out + res)

class TCN(nn.Module):
	def __init__(self, in_n, ch_n, kernel_size = 2, dropout = 0.2):
		super(TCN, self).__init__()
		layers = []
		lvl_n = len(ch_n)
		for i in range(lvl_n):
			dilation = 2**i
			ch_in = 