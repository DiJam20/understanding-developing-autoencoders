####
# Adopted from:
# Markos Genios. "Representation Learning with Increasing Dimensionality in Autoencoders".
# Unpublished. Bachelor's Thesis. Goethe University Frankfurt, 2024.
####

## import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys
import argparse
import time
from tqdm import tqdm

from autoencoder import LinearAutoencoder, NonLinearAutoencoder



## define the training function
def train(model,train_loader,optimizer,epoch,device):
	model.to(device)
	model.train()
	train_loss = 0

	pbar = tqdm(enumerate(train_loader),total=len(train_loader))
	all_losses = []
	for batch_idx, (data, target) in pbar:
		data = Variable(data)
		optimizer.zero_grad()
		batch_size = data.size()[0]
		input = data.view(batch_size,-1).to(device)
		encoded, decoded = model(input)
		loss = F.mse_loss(decoded, input)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader),
			loss.item() / len(data)))
		all_losses.append(loss.item())
	train_loss /= len(train_loader)
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss))
	return train_loss, all_losses

def train_vali_all_epochs(model,train_loader,vali_loader,optimizer,n_epochs,device,save_path=None):
	train_losses = []
	vali_losses = []
	all_train_losses = []
	for epoch in range(n_epochs):
		train_loss,train_loss_per_batch = train(model,train_loader,optimizer,epoch,device)
		vali_loss,_,_ = test(model,vali_loader,device)
		train_losses.append(train_loss)
		all_train_losses.append(train_loss_per_batch)
		vali_losses.append(vali_loss)
		if save_path is not None:
			if not os.path.exists(save_path):
				os.makedirs(save_path)
				print('Directory created:', save_path)
			torch.save(model.state_dict(), save_path + 'model_weights_epoch{}.pth'.format(epoch))
			print('Weights saved.')
	np.save(save_path + 'all_train_losses.npy',all_train_losses)
	print('All train losses saved.')
	return train_losses, vali_losses

def size_per_epoch(size_ls,n_epochs,type='step'):
	if type == 'constant':
		size_epoch = [size_ls[0]]*n_epochs
	elif type == 'step':
		nsteps = len(size_ls)
		rep = n_epochs//nsteps
		size_epoch = np.repeat(size_ls,rep)
		if n_epochs - len(size_epoch) > 0:
			size_epoch = np.append(size_epoch,[size_ls[-1]]*(n_epochs - len(size_epoch)))
	return size_epoch

def varying_dim(A, B):
	'''returns the index of the dimension-tuple, at which both numbers differ.'''
	if A.shape == B.shape:
		return -1
	return torch.nonzero(torch.tensor(A.shape) - torch.tensor(B.shape))[0].item()

def set_old_weights(new_weights, trained_weights):
	'''Sets the part, that corresponds in size with the trained_weights-tensor
	to the larger tensor "new_weights".'''
	var_dim = varying_dim(new_weights, trained_weights)
	print('debug var_dim',var_dim)
	if var_dim == 0:
		new_weights[:trained_weights.size(0)] = trained_weights[:trained_weights.size(0)]
	elif var_dim == 1:
		new_weights[:,:trained_weights.size(1)] = trained_weights[:,:trained_weights.size(1)]
	elif var_dim == -1:
		new_weights = trained_weights
	return new_weights

def divide_latent_tensor(tensor, new_dim):
	"""
	Input: 
	tensor ~ latent vector or rows of latent vectors
	new_dim ~ new dimension for the latent vector

	Output:
	new_tensor ~ The latent vector (or the latent vectors in each row)
	get extended to the dimension new_dim, with an integer multiple of the
	original weights. The rest of the new tensor gets sampled values from
	a distribution with mean and std being based on the values of the original weights.
	"""
	one_dim = False
	gpu = False
	if tensor.device.type == 'gpu':  
		gpu = True
	if len(tensor.shape) == 1:  
		one_dim = True
		tensor = tensor.unsqueeze(0)
	tensor_size = tensor.size()
	# Calculate the number of lines and the number of weights in each line.
	lines = tensor_size[0]  
	multiple = new_dim//tensor_size[1]  
	# Calculate the rest of the weights that need to be added.
	rest = new_dim % tensor_size[1] 
	new_tensor = torch.zeros(lines, new_dim)
	# Fill the new tensor with the old tensor as often as it fits.
	new_tensor[:, :multiple*tensor_size[1]] = torch.cat([tensor]*multiple, dim=1)  
	mean = tensor.view(-1).mean()  
	std = tensor.view(-1).std() 
	# add a little noise to the old weights
	new_tensor[:, :multiple*tensor_size[1]] += torch.normal(mean=mean, std=0.1*std, size=(lines, multiple*tensor_size[1]))
	# Fill the rest of the new tensor with random values.
	new_tensor[:, multiple*tensor_size[1]:] = torch.normal(mean=mean, std=std, size=(lines, rest)) 
	if one_dim:
		new_tensor = new_tensor[0]  
	if gpu:
		new_tensor.to('gpu')
	return new_tensor


def extend_tensor_with_noise(tensor, new_dim, dim=0):
    """
	Fills tensor with noise to match new_dim along the specified dimension.
	Noise is drawn from a normal distribution with mean and standard deviation
	based on the original tensor values.
	1D tensors (biases) are extended along dimension 0.
	2D tensors (weights) are extended along dimension 0 for encoder weights
	and along dimension 1 for decoder weights.
	Encoder (dim 0)
		w1,1 w1,2 ... w1,128	-->		w1,1 w1,2 ... w1,128
		...						-->		...
		w4,1 w4,2 ... w4,128 	-->		...
										w4,1 w4,2 ... w4,128
	Decoder (dim 1)
		w1,1 w1,2 ... w1,4		-->		w1,1 w1,2 ... w1,4 ... w1,8
		...						-->		...
		w128,1 w128,2 ... w128,4-->		w128,1 w128,2 ... w128,4 ... w128,8
    """
    # Get tensor shape
    tensor_shape = tensor.shape
    tensor_dims = len(tensor_shape)
    
    # If tensor is 1D and we're trying to extend along second dimension, that's an error
    if tensor_dims == 1 and dim == 1:
        raise ValueError("Cannot extend a 1D tensor along dimension 1")
    
    # If new dimension is not larger, return the original tensor (not a slice)
    if new_dim <= tensor_shape[dim]:
        return tensor
    
    if tensor_dims == 1:  # 1D tensor (bias)
        old_dim = tensor_shape[0]
        
        # Create a new tensor with the desired size
        new_tensor = torch.zeros(new_dim, device=tensor.device)
        
        # Copy the original values
        new_tensor[:old_dim] = tensor
        
        # Calculate mean and standard deviation
        mean = tensor.mean()
        std = tensor.std()
        
        # Fill the rest with noise
        new_tensor[old_dim:] = torch.normal(mean=mean, std=std, size=(new_dim - old_dim,))
        
    elif tensor_dims == 2:  # 2D tensor (weight matrix)
        # Handle extension along different dimensions
        if dim == 0:  # Extend along first dimension (for encoder weights)
            old_dim = tensor_shape[0]
            weight_size = tensor_shape[1]
            
            # Create new tensor
            new_tensor = torch.zeros(new_dim, weight_size, device=tensor.device)
            
            # Copy original values
            new_tensor[:old_dim] = tensor
            
            # Calculate stats
            mean = tensor.mean()
            std = tensor.std()
            
            # Fill rest with noise
            new_tensor[old_dim:] = torch.normal(mean=mean, std=std, size=(new_dim - old_dim, weight_size))
            
        elif dim == 1:  # Extend along second dimension (for decoder weights)
            old_dim = tensor_shape[1]
            weight_size = tensor_shape[0]
            
            # Create new tensor
            new_tensor = torch.zeros(weight_size, new_dim, device=tensor.device)
            
            # Copy original values
            new_tensor[:, :old_dim] = tensor
            
            # Calculate stats
            mean = tensor.mean()
            std = tensor.std()
            
            # Fill rest with noise
            new_tensor[:, old_dim:] = torch.normal(mean=mean, std=std, size=(weight_size, new_dim - old_dim))
    
    # Handle GPU device if needed
    if tensor.device.type == 'gpu':
        new_tensor = new_tensor.to('gpu')
        
    return new_tensor

def divide_enc_tensor(tensor, new_dim):
	"""
	Multiplies the given tensor in a way that it matches the new_dim parameter.
	Weights are just doubled as far as possible (integer multiple).
	The rest of the values are drawn from a gaussian with mean and std based
	on the values of the input tensor.
	"""
	size = tensor.size()
	old_dim, weight_size = size[0], size[1]
	multiple = new_dim//old_dim
	rest = new_dim%old_dim

	# Repeat the original tensor to construct the first 8 lines
	div_tensor = torch.cat([tensor]*multiple, dim=0)

	# Calculate mean and standard deviation of the original tensor
	mean = tensor.view(-1).mean()
	std_dev = tensor.std(dim=1).std()

	# add noise to the new weights
	noise = torch.normal(mean=mean, std=0.1*std_dev, size=(old_dim*(multiple-1), weight_size))
	div_tensor[old_dim:] += noise

	# Generate new samples based on a normal distribution with similar statistics
	rest_samples = torch.normal(mean=mean, std=std_dev, size=(rest, weight_size))
	# Concatenate the new samples to complete the final tensor
	new_weights = torch.cat((div_tensor, rest_samples), dim=0)
	if tensor.device.type == 'gpu':
		new_weights.to('gpu')
	return new_weights

def cut_enc_tensor(tensor):
    cut_number = np.ceil(tensor.shape[0] * 0.1)
    return tensor[:-int(cut_number), :]

def cut_latent_tensor(tensor):
    cut_number = np.ceil(tensor.shape[0] * 0.1)
    return tensor[:-int(cut_number)]

def cut_dec_tensor(tensor):
    cut_number = np.ceil(tensor.shape[1] * 0.1)
    return tensor[:, :-int(cut_number)].clone()  # Klonen, weil sich sonst der "stride" nicht ändert! (Technischer stuff)

def l_develope_AE(new_n_hidden_ls,hyperparam,save_path,epoch,manner='naiv'):
	new_size = new_n_hidden_ls[-1]
	# Create new autoencoder with corresponding bottleneck size.
	n_layers = hyperparam['n_layers']
	autoencoder = LinearAutoencoder(hyperparam['n_input'],new_n_hidden_ls,n_layers)

	# Load the weights learned in the last epoch.
	state_dict = torch.load(save_path + 'model_weights_epoch{}.pth'.format(epoch-1))
	num_weights = state_dict["encoder.encoder_{}.weight".format(n_layers)].size(1)  
	# Use different manners to determine the new weights.
	if manner == 'naiv':
		new_weights_encoder = torch.randn(new_size, num_weights)
		new_bias_encoder = torch.randn(new_size)
		new_weights_decoder = torch.randn(num_weights, new_size)
	elif manner == 'cell_division':
		# In the following three lines, a part of the new, randomly initialized, weights get assigned
		# to the weights that were learned in the last epoch.
		new_weights_encoder = divide_enc_tensor(state_dict["encoder.encoder_{}.weight".format(n_layers)], new_size)
		new_bias_encoder = divide_latent_tensor(state_dict["encoder.encoder_{}.bias".format(n_layers)], new_size)
		new_weights_decoder = divide_latent_tensor(state_dict["decoder.decoder_1.weight"], new_size)
		
	# Use the new weights to update the state_dict.
	state_dict["encoder.encoder_{}.weight".format(n_layers)] = \
		set_old_weights(new_weights_encoder, state_dict["encoder.encoder_{}.weight".format(n_layers)])
	state_dict["encoder.encoder_{}.bias".format(n_layers)] = \
		set_old_weights(new_bias_encoder, state_dict["encoder.encoder_{}.bias".format(n_layers)])
	state_dict["decoder.decoder_1.weight"] = \
		set_old_weights(new_weights_decoder, state_dict["decoder.decoder_1.weight"])
	# use new state_dict to update model weights
	print('debug nan in encoder weights',torch.isnan(state_dict["encoder.encoder_{}.weight".format(n_layers)]).sum())
	print('debug nan in encoder bias',torch.isnan(state_dict["encoder.encoder_{}.bias".format(n_layers)]).sum())
	print('debug nan in decoder weights',torch.isnan(state_dict["decoder.decoder_1.weight"]).sum())
	autoencoder.load_state_dict(state_dict)

	return autoencoder


def nl_develope_AE(new_n_hidden_ls, hyperparam, save_path, epoch, manner='naiv'):
    new_size = new_n_hidden_ls[-1]
    # Create new autoencoder with corresponding bottleneck size.
    n_layers = hyperparam['n_layers']
    autoencoder = NonLinearAutoencoder(hyperparam['n_input'], new_n_hidden_ls, n_layers)

    # Load the weights learned in the last epoch.
    state_dict = torch.load(save_path + 'model_weights_epoch{}.pth'.format(epoch-1))
    num_weights = state_dict["encoder.encoder_{}.weight".format(n_layers)].size(1)  
    # Use different manners to determine the new weights.
    if manner == 'naiv':
        new_weights_encoder = torch.randn(new_size, num_weights)
        new_bias_encoder = torch.randn(new_size)
        new_weights_decoder = torch.randn(num_weights, new_size)
    elif manner == 'cell_division':
        # In the following three lines, a part of the new, randomly initialized, weights get assigned
        # to the weights that were learned in the last epoch.
        new_weights_encoder = divide_enc_tensor(state_dict["encoder.encoder_{}.weight".format(n_layers)], new_size)
        new_bias_encoder = divide_latent_tensor(state_dict["encoder.encoder_{}.bias".format(n_layers)], new_size)
        new_weights_decoder = divide_latent_tensor(state_dict["decoder.decoder_1.weight"], new_size)
    elif manner == 'gaussian':
        # Important: For encoder weights and bias, extend along dimension 0
        # For decoder weights, extend along dimension 1
        new_weights_encoder = extend_tensor_with_noise(state_dict["encoder.encoder_{}.weight".format(n_layers)], new_size, dim=0)
        new_bias_encoder = extend_tensor_with_noise(state_dict["encoder.encoder_{}.bias".format(n_layers)], new_size, dim=0)
        new_weights_decoder = extend_tensor_with_noise(state_dict["decoder.decoder_1.weight"], new_size, dim=1)
        
    # Use the new weights to update the state_dict.
    state_dict["encoder.encoder_{}.weight".format(n_layers)] = new_weights_encoder
    state_dict["encoder.encoder_{}.bias".format(n_layers)] = new_bias_encoder
    state_dict["decoder.decoder_1.weight"] = new_weights_decoder
    
    # Debug checks for NaN values
    print('debug nan in encoder weights', torch.isnan(state_dict["encoder.encoder_{}.weight".format(n_layers)]).sum())
    print('debug nan in encoder bias', torch.isnan(state_dict["encoder.encoder_{}.bias".format(n_layers)]).sum())
    print('debug nan in decoder weights', torch.isnan(state_dict["decoder.decoder_1.weight"]).sum())
    
    # Load the state dict into the model
    autoencoder.load_state_dict(state_dict)

    return autoencoder
	

def l_dev_train_vali_all_epochs(model,size_ls,manner,train_loader,vali_loader,optimizer,n_epochs,device,save_path=None):
	if save_path is None:
		save_path = './'
	else:
		if not os.path.exists(save_path):
			os.makedirs(save_path)
			print('Directory created:', save_path)
	train_losses = []
	all_train_losses = []
	vali_losses = []
	size_each_epoch = size_per_epoch(size_ls,n_epochs,type='step')
	np.save(save_path + 'size_each_epoch.npy',size_each_epoch)
	hyperparam = model.get_hyperparams()
	print(size_each_epoch)

	for epoch in range(n_epochs):
		print(size_each_epoch[epoch])
		# Create new autoencoder with corresponding bottleneck size.
		new_n_hidden_ls = np.append(hyperparam['n_hidden_ls'][:-1] , size_each_epoch[epoch])
		if epoch == 0:
			ae =LinearAutoencoder(hyperparam['n_input'],new_n_hidden_ls,hyperparam['n_layers'])

		else:
			ae = l_develope_AE(new_n_hidden_ls,hyperparam,save_path=save_path,epoch=epoch,manner=manner)
			optimizer = torch.optim.SGD(ae.parameters(),lr=1e-1,momentum=0.9)
			
		train_loss,train_loss_per_batch = train(ae,train_loader,optimizer,epoch,device)
		vali_loss,_,_ = test(ae,vali_loader,device)
		train_losses.append(train_loss)
		all_train_losses.append(train_loss_per_batch)
		vali_losses.append(vali_loss)

		# Save the weights of the last epoch.
		torch.save(ae.state_dict(), save_path + 'model_weights_epoch{}.pth'.format(epoch))
		print('Weights saved.')
	np.save(save_path + 'all_train_losses.npy',all_train_losses)
	print('All train losses saved.')
	return train_losses, vali_losses

def nl_dev_train_vali_all_epochs(model,size_ls,manner,train_loader,vali_loader,optimizer,n_epochs,device,save_path=None):
	if save_path is None:
		save_path = './'
	else:
		if not os.path.exists(save_path):
			os.makedirs(save_path)
			print('Directory created:', save_path)
	train_losses = []
	all_train_losses = []
	vali_losses = []
	size_each_epoch = size_per_epoch(size_ls,n_epochs,type='step')
	np.save(save_path + 'size_each_epoch.npy',size_each_epoch)
	hyperparam = model.get_hyperparams()
	print(size_each_epoch)

	current_bottleneck_size = None
	ae = None

	for epoch in range(n_epochs):
		print(size_each_epoch[epoch])
		new_n_hidden_ls = np.append(hyperparam['n_hidden_ls'][:-1] , size_each_epoch[epoch])
		
		# Only reinitialise autoencoder and optimiser when the bottleneck size increases
		if size_each_epoch[epoch] != current_bottleneck_size or ae is None:
			current_bottleneck_size = size_each_epoch[epoch]

			if epoch == 0:
				ae = NonLinearAutoencoder(hyperparam['n_input'], new_n_hidden_ls, hyperparam['n_layers'])
			else:
				ae = nl_develope_AE(new_n_hidden_ls, hyperparam, save_path=save_path, epoch=epoch, manner=manner)

			optimizer = torch.optim.SGD(ae.parameters(), lr=1e-1, momentum=0.9)
		
		train_loss,train_loss_per_batch = train(ae,train_loader,optimizer,epoch,device)
		vali_loss,_,_ = test(ae,vali_loader,device)
		train_losses.append(train_loss)
		all_train_losses.append(train_loss_per_batch)
		vali_losses.append(vali_loss)

		# Save the weights of the last epoch.
		torch.save(ae.state_dict(), save_path + 'model_weights_epoch{}.pth'.format(epoch))
		print('Weights saved.')
	np.save(save_path + 'all_train_losses.npy',all_train_losses)
	print('All train losses saved.')
	return train_losses, vali_losses

# def dev_train_vali_converge(model, size_ls, manner, train_loader, vali_loader, optimizer, device, save_path=None):
# 	if save_path is None:
# 		save_path = './'
# 	else:
# 		if not os.path.exists(save_path):
# 			os.makedirs(save_path)
# 			print('Directory created:', save_path)
# 	train_losses = []
# 	all_train_losses = []
# 	vali_losses = []
# 	hyperparam = model.get_hyperparams()
# 	epoch = 0
	
# 	for size in size_ls:
# 		new_n_hidden_ls = np.append(hyperparam['n_hidden_ls'][:-1], size)
		
# 		converged = False
# 		train_loss_old = float('inf')

# 		while not converged:
# 			print(new_n_hidden_ls[-1])
# 			if epoch == 0:
# 				ae = LinearAutoencoder(hyperparam['n_input'], new_n_hidden_ls, hyperparam['n_layers'])
# 			else:
# 				ae = develope_AE(new_n_hidden_ls, hyperparam, save_path=save_path, epoch=epoch, manner=manner)
# 				optimizer = torch.optim.SGD(ae.parameters(), lr=1e-1, momentum=0.9)

# 			train_loss, train_loss_per_batch = train(ae, train_loader, optimizer, epoch, device)
# 			vali_loss, _, _ = test(ae, vali_loader, device)
# 			train_losses.append(train_loss)
# 			all_train_losses.append(train_loss_per_batch)
# 			vali_losses.append(vali_loss)

# 			# Save the weights of the current epoch.
# 			torch.save(ae.state_dict(), save_path + 'model_weights_epoch{}.pth'.format(epoch))

# 			# Check for convergence
# 			if new_n_hidden_ls[-1] == size_ls[-1]:
# 				if train_loss < train_loss_old * 0.999:
# 					epoch += 1
# 					train_loss_old = train_loss
# 				else:
# 					epoch += 1
# 					converged = True
# 			else:
# 				if train_loss < train_loss_old * 0.99:
# 					epoch += 1
# 					train_loss_old = train_loss
# 				else:
# 					epoch += 1
# 					converged = True

# 	np.save(save_path + 'all_train_losses.npy', all_train_losses)
# 	return train_losses, vali_losses


## define the testing function
def test(model,test_loader,device):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for data, _ in test_loader:
			data = Variable(data, volatile=True).to(device)

			input = data.view(data.size(0), -1)
			encoded, decoded = model(input)
			test_loss += F.mse_loss(decoded, input).item()
		test_loss /= len(test_loader)
		print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss, decoded, data


## define training function for convolutional autoencoder
def train_conv(model,train_loader,optimizer,epoch):
	model.train()
	train_loss = 0

	pbar = tqdm(enumerate(train_loader),total=len(train_loader))
	for batch_idx, (data, target) in pbar:
		data = Variable(data)
		optimizer.zero_grad()
		# print('debug data shape',data.shape)
		encoded, decoded = model(data)
		# print('debug decoded shape',decoded.shape)
		# print('debug encoded shape',encoded.shape)
		loss = F.mse_loss(decoded, data)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader),
			loss.item() / len(data)))
	train_loss /= len(train_loader)
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss))
	return train_loss

## define testing function for convolutional autoencoder
def test_conv(model,test_loader):
	model.eval()
	test_loss = 0
	for data, _ in test_loader:
		data = Variable(data, volatile=True)
		encoded, decoded = model(data)
		test_loss += F.mse_loss(decoded, data, size_average=False).item()
	test_loss /= len(test_loader)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss, decoded,data


def train_vae(model,train_loader,optimizer,epoch):
	model.train()
	train_loss = 0

	pbar = tqdm(enumerate(train_loader),total=len(train_loader))
	for batch_idx, (data, target) in pbar:
		data = Variable(data)
		optimizer.zero_grad()
		# print('debug data shape',data.shape)
		batch_size = data.size()[0]
		input = data.view(batch_size,-1)
		decoded, mu, logvar = model(input)
		# print('debug decoded shape',decoded.shape)
		# print('debug encoded shape',encoded.shape)
		loss = F.mse_loss(decoded, input)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader),
			loss.item() / len(data)))
	train_loss /= len(train_loader)
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss))
	return train_loss

def test_vae(model,test_loader,device='cpu'):
	model.eval()
	test_loss = 0
	for data, _ in test_loader:
		with torch.no_grad():
			input = data.view(data.size(0), -1).to(device)
			decoded, mu, logvar = model(input)
			test_loss += F.mse_loss(decoded, input, size_average=False).item()
	test_loss /= len(test_loader)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss, decoded,data