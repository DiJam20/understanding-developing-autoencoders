import os
import numpy as np
import torch
from tqdm import tqdm
import act_max_util as amu
from autoencoder import *
from solver import *
from model_utils import *


steps = 100               # perform 100 iterations
alpha = torch.tensor(100) # learning rate (step size)
verbose = False           # print activation every step
L2_Decay = True           # enable L2 decay regularizer
Gaussian_Blur = False     # enable Gaussian regularizer
Norm_Crop = False         # enable norm regularizer
Contrib_Crop = False      # enable contribution regularizer


def compute_rf_for_single_model(model_idx: int, model_type: str, dataset: str = "mnist", 
                               size_ls: list = None, num_epochs: int = 60,
                               base_path: str = "/home/david/") -> list:
    """
    Compute the receptive fields for all bottleneck neurons over epochs for a single model.

    Args:
        model_idx (int): Index of the model to compute RFs for.
        model_type (str): Type of model to compute RFs for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        size_ls (list): List of sizes for each epoch for DAE models.
        num_epochs (int): Number of epochs to compute RFs for.
    
    Returns:
        rf_matrix (list): List of RFs for all bottleneck
            neurons over all epochs for a single model.
    """
    if dataset.lower() == "mnist":
        IMG_WIDTH, IMG_HEIGHT = 28, 28
        MAX_NEURONS = 32
    elif dataset.lower() == "cifar":
        IMG_WIDTH, IMG_HEIGHT = 32, 32
        MAX_NEURONS = 128
    
    rf_matrix = []
    
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx}", leave=False):
        if dataset.lower() == "mnist":
            data = torch.randn(IMG_WIDTH, IMG_HEIGHT)
            data = data.unsqueeze(0)
            input = data.view(data.size(0), -1)
        else:
            data = torch.randn(3, IMG_WIDTH, IMG_HEIGHT)
            input = data.unsqueeze(0)
        
        input.requires_grad_(True)

        if dataset.lower() == "mnist":
            model_path = f"{base_path}mnist_models/{model_type}/{model_idx}"
            ae = load_model(model_path, model_type, epoch)
        else:
            model_path = f"{base_path}cifar_models/{model_type}/{model_idx}"
            ae = load_conv_model(model_path, model_type=model_type, epoch=epoch, size_ls=size_ls)
        
        layer_name = 'bottle_neck'
        activation_dictionary = {}

        if dataset.lower() == "mnist":
            ae.encoder.encoder_3.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))
        else:
            ae.encoder[11].register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

        loop_size = size_ls[epoch] if model_type == "dae" else MAX_NEURONS

        rf_ls = []
        for i in range(loop_size):
            output = amu.act_max(network=ae,
                            input=input,
                            layer_activation=activation_dictionary,
                            layer_name=layer_name,
                            unit=i,
                            steps=steps,
                            alpha=alpha,
                            verbose=verbose,
                            L2_Decay=L2_Decay,
                            Gaussian_Blur=Gaussian_Blur,
                            Norm_Crop=Norm_Crop,
                            Contrib_Crop=Contrib_Crop,
                            )
            rf_ls.append(output.detach().numpy())
        
        max_size = MAX_NEURONS

        # Pad the rf_ls list to max_size before converting to numpy array
        if loop_size < max_size:
            zero_rf = np.zeros_like(rf_ls[0])
            for _ in range(loop_size, max_size):
                rf_ls.append(zero_rf.copy())
        
        rf_ls = np.array(rf_ls).squeeze()
        rf_matrix.append(rf_ls)

    return rf_matrix


def compute_rfs(model_type: str, dataset: str = "mnist", 
               size_ls: list = None, num_models: int = 10, 
               num_epochs: int = 60, base_path="/home/david/") -> None:
    """
    Compute the receptive fields of bottleneck neurons for all models over all epochs.
    
    Args:
        model_type (str): Type of model to compute RFs for.
        dataset (str): Dataset used for training ('mnist' or 'cifar').
        size_ls (list): List of sizes for each epoch for DAE models.
        num_models (int): Number of models to compute RFs for.
        num_epochs (int): Number of epochs to compute RFs for.
        
    Returns:
        None: Saves the RFs to a file.
    """
    result_file = f"Results/{dataset}_{model_type}_rfs.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return np.load(result_file)

    # For all different models
    rf_matrices = []

    for model_idx in tqdm(range(num_models), desc=f"{model_type.upper()}"):
        rf_matrix = compute_rf_for_single_model(
            model_idx, 
            model_type, 
            dataset=dataset,
            size_ls=size_ls,
            num_epochs=num_epochs,
            base_path=base_path
        )

        rf_matrices.append(rf_matrix)

    np.save(result_file, rf_matrices)
    
    return rf_matrices