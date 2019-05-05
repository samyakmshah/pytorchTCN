import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
# sys.path.append("") # Append path here if you need to
from utils import data_generator
from utils import rec_field
from model import TCN
import numpy as np
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser(description = "TCN Sequential MNIST")
# parser.add_argument('--batch_size', type = int, default = 64, metavar = 'N', help='batch size (default = 64)')
# parser.add_argument('--cuda', action='store_false', help='use CUDA arrays for GPU (default = True)')
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout for convolutional layers (default = 0.05)')
# parser.add_argument('--clip', type = float, default=-1, help='gradient clipping (default = -1)')
# parser.add_argument('--epochs', type = int, default=20, help='number epochs during training (default = 20)')
# parser.add_argument('--ksize', type = int, default=7, help='kernel size (default = 7)')
# parser.add_argument('--levels', type = int, default=8, help='number of levels in TCN (default = 8)')
# parser.add_argument('')
