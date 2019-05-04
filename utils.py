import torch
from torchvision import datasets, transforms


def data_generator(url_root, batch_size):
    # Mean and standard deviation of MNIST digits
    trans = transforms.compose([transforms.toTensor(), transforms.Normalize((0.1307), (0.3081))])

	train_set = datasets.MNIST(root=url_root, train=True, download=True, transform = trans)
	test_set = datasets.MNIST(root=url_root, train=True, download = True, transform = trans)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)

	return train_loader, test_loader