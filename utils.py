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

def rec_field(d, kernel_size):

	d = residual_block_calc(d)
	k = residual_block_calc(k)

    if len(d) == 0:
        return 1
    
    d_cur = d[-1]
    kernel_size_cur = kernel_size[-1]
    d_new = d[:-1]
    kernel_size_new = kernel_size[:-1]
    
    return ((kernel_size_cur-1)*d_cur)+1 + rec_field(d_new, kernel_size_new) - 1
    
def residual_block_calc(d):
    d = [[i, i] for i in d]
    
    flat_list = []
    for sublist in d:
        for item in sublist:
            flat_list.append(item)
    return flat_list
