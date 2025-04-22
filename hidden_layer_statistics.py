import os

import numpy as np
import torch
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


nl_layers_to_measure = [
    'encoder_activation_1',
    'encoder_activation_2',
    'encoder_activation_3',
    'decoder_activation_1',
    'decoder_activation_2'
]

conv_layers_to_measure = [
    'encoder_0',
    'encoder_2',
    'encoder_4',
    'encoder_6',
    'encoder_8',
    'encoder_10',
    'linear_0',
    'decoder_0',
    'decoder_2',
    'decoder_4',
    'decoder_6',
    'decoder_8'
]


def get_model_activations(model, image: torch.Tensor, dataset) -> list:
    """
    Get activations for a model given an image.
    
    Args:
        model: Autoencoder model
        image: Image to evaluate
        dataset: Dataset ('mnist' or 'cifar')
        
    Returns:
        list: List of activations for each layer
    """
    with torch.no_grad():
        if dataset == 'mnist':
            _, _, activations = model.forward(image, return_activations=True)
            
            layer_activations = []
            for layer in nl_layers_to_measure:
                act = activations[layer]
                layer_activations.append(act.detach().cpu())
        
        elif dataset == 'cifar':
            if image.dim() == 1:
                # Reshape [3072] to [1, 3, 32, 32]
                image = image.reshape(1, 3, 32, 32)
                
            _, _, activations = model.forward(image, return_activations=True)
            
            layer_activations = []
            for layer in conv_layers_to_measure:
                act = activations[layer]
                # Flatten the activation from convolutional layers
                if act.dim() > 2:
                    act = act.flatten(1)
                layer_activations.append(act.detach().cpu())
    
    return layer_activations
    

def evaluate_model_activations(tensor_test_images: torch.Tensor, ae: NonLinearAutoencoder,
                               dataset: str) -> list:
    """
    Evaluate per-neuron activations for a model across test images.

    Args:
        tensor_test_images: List of test images to be given to the model
        ae: Autoencoder model
        dataset: Dataset ('mnist' or 'cifar')

    Returns:
        list: List of activation statistics for each layer
    """
    if dataset == 'mnist':
        layers_to_measure = nl_layers_to_measure
    elif dataset == 'cifar':
        layers_to_measure = conv_layers_to_measure

    all_layer_acts = [[] for _ in range(len(layers_to_measure))]
    
    # Process all images
    for img in tensor_test_images:
        layer_acts = get_model_activations(ae, img, dataset)
        
        # Store activations for each layer
        for i, acts in enumerate(layer_acts):
            all_layer_acts[i].append(acts)
    
    # Compute statistics for each layer
    results = []
    for i, layer_acts in enumerate(all_layer_acts):
        acts_tensor = torch.stack(layer_acts)

        # If acts_tensor is multidimensional flatten it
        if acts_tensor.dim() > 2:
            # Keep only batch dimension
            acts_tensor = acts_tensor.flatten(1)
        
        always_zero_mask = torch.all(acts_tensor == 0, dim=0)
        never_zero_mask = torch.all(acts_tensor != 0, dim=0)
        
        # Count and calculate percentages
        total_neurons = acts_tensor.shape[1]
        always_zero_count = always_zero_mask.sum().item()
        never_zero_count = never_zero_mask.sum().item()
        sometimes_zero_count = total_neurons - always_zero_count - never_zero_count
        
        # Initialize arrays for statistics
        mean_acts = np.zeros(total_neurons)
        std_acts = np.zeros(total_neurons)
        
        # Calculate statistics for each neuron
        for j in range(total_neurons):
            if not always_zero_mask[j]:
                # For sometimes active neurons, calculate mean and std including zeros
                mean_acts[j] = acts_tensor[:, j].mean().item()
                std_acts[j] = acts_tensor[:, j].std().item()
                # For always zero neurons, mean and std remain 0
        
        # Calculate zero percentages per neuron
        zero_tensor = (acts_tensor == 0).float()
        mean_zeros = torch.mean(zero_tensor, dim=0).numpy() * 100
        std_zeros = torch.std(zero_tensor, dim=0).numpy() * 100
        
        # Create arrays to store the category percentages for each neuron
        always_zero_arr = np.zeros(total_neurons)
        never_zero_arr = np.zeros(total_neurons)
        sometimes_zero_arr = np.zeros(total_neurons)
        
        # Fill arrays based on masks
        always_zero_arr[always_zero_mask.cpu().numpy()] = 100
        never_zero_arr[never_zero_mask.cpu().numpy()] = 100
        sometimes_zero_arr[~(always_zero_mask | never_zero_mask).cpu().numpy()] = 100
        
        # Stack all statistics
        layer_stats = np.stack([
            mean_acts, std_acts, mean_zeros, std_zeros,
            always_zero_arr, never_zero_arr, sometimes_zero_arr
        ])
        
        results.append(layer_stats)
    
    return results


def compute_activation_for_single_model(model_idx: int, model_type: str, 
                                        test_images: list, dataset: str, 
                                        epoch: int, base_path:str) -> list:
    """
    Compute activations for a single model across all epochs.

    Args:
        model_type: Type of model ('sae' or 'dae')
        test_images: List of test images to be given to the model
        dataset: Dataset ('mnist' or 'cifar')
        num_epochs: Number of epochs to process
        epoch: Epoch to evaluate

    Returns:
        list: Activation matrix for the model
    """

    if dataset == 'mnist':
        layers_to_measure = nl_layers_to_measure
        neurons_per_layer = [512, 128, 32, 128, 512]
    elif dataset == 'cifar':
        layers_to_measure = conv_layers_to_measure
        neurons_per_layer = [
            32*16*16,
            32*16*16,
            64*8*8,
            64*8*8,
            64*4*4,
            128,
            64*4*4,
            64*8*8,
            64*8*8,
            32*16*16,
            32*16*16,
            3*32*32
        ]
    
    num_layers = len(layers_to_measure)

    results = np.zeros((num_layers, 7, max(neurons_per_layer)))
    results.fill(np.nan)
    
    if dataset == 'mnist':
        model = load_model(f'{base_path}mnist_models/{model_type}/{model_idx}', model_type, epoch)
    elif dataset == 'cifar':
        if model_type == 'sae':
            model = load_conv_model(f'{base_path}cifar_models/{model_type}/{model_idx}', 
                                  model_type, epoch)
        else:
            model = load_conv_model(f'{base_path}cifar_models/{model_type}/{model_idx}', 
                                  model_type, epoch, [128] * (epoch + 1))
    
    layer_results = evaluate_model_activations(test_images, model, dataset)
    
    # Store results for each layer
    for layer_idx, layer_data in enumerate(layer_results):
        num_neurons = layer_data.shape[1]
        results[layer_idx, :, :num_neurons] = layer_data
    
    return results


def compute_activation_over_epochs_for_single_model(model_idx: int, model_type: str, 
                                        test_images: list, dataset: str, 
                                        num_epochs: int, epoch: int, base_path:str) -> list:
    """
    Compute activations for a single model across all epochs.

    Args:
        model_type: Type of model ('sae' or 'dae')
        test_images: List of test images to be given to the model
        dataset: Dataset ('mnist' or 'cifar')
        num_epochs: Number of epochs to process
        epoch: Epoch to evaluate

    Returns:
        list: Activation matrix for the model
    """

    if dataset == 'mnist':
        layers_to_measure = nl_layers_to_measure
        neurons_per_layer = [512, 128, 32, 128, 512]
    elif dataset == 'cifar':
        layers_to_measure = conv_layers_to_measure
        neurons_per_layer = [
            32*16*16,
            32*16*16,
            64*8*8,
            64*8*8,
            64*4*4,
            128,
            64*4*4,
            64*8*8,
            64*8*8,
            32*16*16,
            32*16*16,
            3*32*32
        ]
    
    num_layers = len(layers_to_measure)
    
    results = np.zeros((num_epochs, num_layers, 7, max(neurons_per_layer)))
    results.fill(np.nan)
    
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        ae = load_model(f'{base_path}mnist_models/{model_type}/{model_idx}', model_type, epoch)
        layer_results = evaluate_model_activations(test_images, ae)
        
        # Store results for each layer
        for layer_idx, layer_data in enumerate(layer_results):
            num_neurons = layer_data.shape[1]
            results[epoch, layer_idx, :, :num_neurons] = layer_data
    
    return results


def compute_neuron_activations(model_type, dataset, num_models=40, num_epochs=10, epoch=59):
    """
    Compute activations for all models at a specific epoch.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset name ('mnist' or 'cifar')
        num_models: Number of models to process
        epoch: Specific epoch to analyze (default: 59)
    
    Returns:
        None: Saves activation data for all models
    """
    result_file = f'Results/hidden_layer_act_{model_type}_{dataset}.npy'
    
    # Check if results already exist
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    if dataset == 'mnist':
        test_images, _ = load_mnist_tensor()
        layers_to_measure = nl_layers_to_measure
        neurons_per_layer = [512, 128, 32, 128, 512]
    elif dataset == 'cifar':
        test_images, _ = load_cifar_tensor()
        layers_to_measure = conv_layers_to_measure
        neurons_per_layer = [
            32*16*16,
            32*16*16,
            64*8*8,
            64*8*8,
            64*4*4,
            128,
            64*4*4,
            64*8*8,
            64*8*8,
            32*16*16,
            32*16*16,
            3*32*32
        ]
    
    all_results = np.zeros((num_models, len(layers_to_measure), 7, max(neurons_per_layer)))
    all_results.fill(np.nan)
    
    for model_idx in tqdm(range(num_models), desc="Processing models"):
        model_results = compute_activation_for_single_model(
            model_idx, model_type, test_images, dataset, epoch, base_path="/home/david/"
        )
        
        all_results[model_idx] = model_results
    
    # Save results
    np.save(result_file, all_results)
    
    return None


def compute_hidden_layer_activation(model_type, dataset='mnist', 
                                   num_models=40, num_epochs = 10, epoch=59):
    """
    Compute activation statistics for all models at a specific epoch.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset name ('mnist' or 'cifar')
        num_models: Number of models to process
        epoch: Specific epoch to analyze
        
    Results:
        None: Saves results to file
    """
    compute_neuron_activations(model_type, dataset, num_models, num_epochs, epoch)
    return None