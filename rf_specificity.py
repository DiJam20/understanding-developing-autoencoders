import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *


def compute_activation_for_single_model(model_idx: int, model_type: str, rf_matrix: np.ndarray,
                                      dataset: str = "mnist", size_ls: list = None, 
                                      num_epochs: int = 60, base_path = "/home/david/") -> list:
    """
    Compute activations for a single model across all epochs.
    
    Args:
        model_idx: Index of the model
        model_type: Type of model ('sae' or 'dae')
        rf_matrix: Pre-loaded receptive fields for this model
        dataset: Dataset used for training ('mnist' or 'cifar')
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
        
    Returns:
        list: Activation matrix for the model
    """
    if dataset.lower() == "mnist":
        MAX_NEURONS = 32
    elif dataset.lower() == "cifar":
        MAX_NEURONS = 128
    
    activation_matrix = []
    
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        if dataset.lower() == "mnist":
            ae = load_model(f'{base_path}mnist_models/{model_type}/{model_idx}', model_type, epoch)
        else:
            ae = load_conv_model(f"{base_path}cifar_models{model_type}/{model_idx}", model_type=model_type, epoch=epoch, size_ls=size_ls)
        
        num_neurons = size_ls[epoch] if model_type == "dae" else MAX_NEURONS
        max_size = MAX_NEURONS
        
        # Get RFs for current epoch
        rf_ls = rf_matrix[epoch][:num_neurons]
        
        # Process each RF
        epoch_activations = []
        with torch.no_grad():
            if dataset.lower() == "mnist":
                inputs = torch.tensor(np.stack(rf_ls), dtype=torch.float32).reshape(len(rf_ls), -1)
            else:
                inputs = torch.tensor(np.stack(rf_ls), dtype=torch.float32)
            
            encoded, _ = ae(inputs)
            absolute_encoded = torch.abs(encoded)
            epoch_activations = absolute_encoded.cpu().numpy()
        
        # Pad activations to maximum size
        padded_activations = np.zeros((max_size, max_size))

        for i, act in enumerate(epoch_activations):
            if i < max_size:
                # Each activation is padded to max_size length
                act_padded = np.zeros(max_size)
                act_padded[:len(act)] = act[:max_size]
                padded_activations[i] = act_padded
        
        activation_matrix.append(padded_activations)
    
    return activation_matrix


def compute_neuron_activations(model_type: str, dataset: str = "mnist", 
                              num_models: int = 40, size_ls: list = None, num_epochs: int = 60,
                              base_path:str = "/home/david/") -> None:
    """
    Compute activations for all models across all epochs.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset used for training ('mnist' or 'cifar')
        num_models: Number of models to process
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
    
    Returns:
        None: Saves activation matrices for all models
    """
    result_file = f'Results/{dataset}_{model_type}_rf_neuron_activations.npy'
    
    # Check if results already exist
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    print(f"Loading receptive fields from {result_file}")
    rf_matrices = np.load(f"Results/{dataset}_{model_type}_rfs.npy", allow_pickle=True)
    
    activation_matrices = []
    
    for model_idx in tqdm(range(num_models), desc=f"Processing models"):
        activation_matrix = compute_activation_for_single_model(
            model_idx,
            model_type,
            rf_matrices[model_idx],
            dataset=dataset,
            size_ls=size_ls,
            num_epochs=num_epochs,
            base_path=base_path
        )
        activation_matrices.append(activation_matrix)
    
    np.save(result_file, np.array(activation_matrices))
    return None


def plot_neuron_activations(model_type: str, dataset: str = "mnist", 
                           epoch: int = 59, neuron_idx: int = 0) -> None:
    """
    Plot activations given a specific RF and epoch (averaged over all models).
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset used for training ('mnist' or 'cifar')
        epoch: Epoch to plot
        neuron_idx: Index of the neuron to plot
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{dataset}_{model_type}_rf_neuron_activations.npy")
    epoch_activations = neuron_activations[:, epoch, neuron_idx, :]

    mean_activations = np.mean(epoch_activations, axis=0)
    std_activations = np.std(epoch_activations, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_activations, color='#1a7adb')
    plt.fill_between(np.arange(len(mean_activations)),
                     mean_activations - std_activations,
                     mean_activations + std_activations,
                     color='#1a7adb', alpha=0.2)
    plt.title(f"Activations given {dataset.upper()} Receptive Field {neuron_idx} (Epoch: {epoch})")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Magnitude")
    
    plt.savefig(f"Results/figures/png/{dataset}_{model_type}_rf_neuron_activations_neurons_{neuron_idx}_epoch_{epoch}.png", dpi=300)
    plt.close()

    return None


def plot_neuron_activations_all_epochs(model_type: str, dataset: str = "mnist", neuron_idx: int = 0) -> None:
    """
    Plot the neuron activation given a specific RF across all epochs.
    Shows the development of how the RF as well as the neurons change over time.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset used for training ('mnist' or 'cifar')
        neuron_idx: Index of the neuron whose RF is used
    
    Returns:
        None: Saves plot to file
    """
    neuron_activations = np.load(f"Results/{dataset}_{model_type}_rf_neuron_activations.npy")
    num_epochs = neuron_activations.shape[1]
    total_neurons = neuron_activations.shape[3]
    
    plt.figure(figsize=(15, 10))
    
    blues = plt.cm.Blues(np.linspace(0.3, 1.0, num_epochs))
    
    # Plot the activation across epochs
    for epoch in range(num_epochs):
        avg_activations = np.mean(neuron_activations[:, epoch, neuron_idx, :], axis=0)
        
        # Darker colors for later neurons
        plt.plot(range(total_neurons), avg_activations, color=blues[epoch], 
                 label=f'Epoch {epoch+1}' if epoch % 10 == 0 or epoch == num_epochs-1 else "")
    
    plt.title(f"Activations in response to RF {neuron_idx} ({dataset.upper()})")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Magnitude")
    
    plt.savefig(f"Results/figures/png/{dataset}_{model_type}_epochs_activations_across_neurons_for_rf_{neuron_idx}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_{model_type}_epochs_activations_across_neurons_for_rf_{neuron_idx}.svg", bbox_inches='tight')
    plt.close()
    
    return None


def compute_rf_specificity(model_type: str, dataset: str = "mnist", 
                          num_models: int = 40, size_ls: list = None, 
                          num_epochs: int = 60, base_path:str = "/home/david/") -> None:
    """
    Compute RF specificity for all models.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset used for training ('mnist' or 'cifar')
        num_models: Number of models to process
        size_ls: List of sizes for DAE models
        num_epochs: Number of epochs to process
    Results:
        None: Saves results to file
    """
    compute_neuron_activations(model_type, dataset, num_models=num_models, 
                             size_ls=size_ls, num_epochs=num_epochs, base_path=base_path)
    
    plot_neuron_activations(model_type, dataset, epoch=num_epochs-1, neuron_idx=0)

# compute_rf_specificity("sae", "mnist", num_models=2, size_ls=None, num_epochs=60)