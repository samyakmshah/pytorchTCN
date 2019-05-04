import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ResidualBlock(nn.Module):
	def __init__(self, ch_in, ch_out, kernel_size, stride, dilation, left_pad, dropout = 0.2):
		super(ResidualBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(ch_in, ch_out, kern_size, stride = stride, padding = (left_pad, 0), dilation = dil))
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)
		self.conv2 = weight_norm(nn.Conv1d(ch_in, ch_out, kern_size, stride = stride, padding = (left_pad, 0), dilation = dil))
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)