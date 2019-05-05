import torch.nn.functional as F
from torch import nn
from tcn import TCN_base

class TCN(nn.Module):
	def __init__(self, in_n, out_n, ch_n, kernel_size, dropout):
		super(TCN, self).__init__()
		self.tcn = TCN_base(in_n, ch_n, kernel_size, dropout)
		self.linear = nn.Linear(ch_n[-1], out_n)

	def forward(self, x):
		y = self.tcn(x) # (Batch, Channels, Length)
		out = self.linear(y[:, :, -1]) # take the last output
		return F.log_softmax(out, dim = 1)



