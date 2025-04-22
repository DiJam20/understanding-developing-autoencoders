import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch

from autoencoder import *
from model_utils import *
from solver import *

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11
})


def add_noise_and_reconstruct(test_images, noise_scale, dataset="mnist", n_components=None, start_noise_idx=0, end_noise_idx=4):
    """
    Add noise to specific principal components of the images and reconstruct.
    
    Args:
        test_images: Test images
        noise_scale: Scale of the noise to add
        dataset: Dataset ('mnist' or 'cifar')
        n_components: Number of PCA components (default: None, will be set based on dataset)
        start_noise_idx: Start index of PCs to add noise to
        end_noise_idx: End index of PCs to add noise to
        
    Returns:
        Reconstructed images with noise added to specified PCs
    """
    # Set dataset-specific parameters
    if dataset.lower() == "mnist":
        if n_components is None:
            n_components = 32
        # For MNIST, images are already flat vectors
        pca_input = test_images
        reshape_shape = len(test_images), 784
    elif dataset.lower() == "cifar":
        if n_components is None:
            n_components = 128
        # For CIFAR, reshape from (N, 3, 32, 32) to (N, 3072)
        pca_input = test_images.reshape(len(test_images), -1)
        reshape_shape = len(test_images), 3, 32, 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'mnist' or 'cifar'.")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_reduced = pca.fit_transform(pca_input)
    
    # Add different noise to each PCA reduced image to specified PCs
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=(len(pca_reduced), end_noise_idx-start_noise_idx))
    pca_reduced[:, start_noise_idx:end_noise_idx] += noise
    
    # Reconstruct
    return pca.inverse_transform(pca_reduced).reshape(reshape_shape)


def add_zeroing_and_reconstruct(test_images, dataset="mnist", n_components=None, keep_start_idx=0, keep_end_idx=4):
    """
    Keep only one PC group and set all others to zero, then reconstruct.
    
    Args:
        test_images: Test images
        dataset: Dataset ('mnist' or 'cifar')
        n_components: Number of PCA components (default: None, will be set based on dataset)
        keep_start_idx: Start index of PCs to keep
        keep_end_idx: End index of PCs to keep
        
    Returns:
        Reconstructed images with only specified PCs kept
    """
    # Set dataset-specific parameters
    if dataset.lower() == "mnist":
        if n_components is None:
            n_components = 32
        # For MNIST, images are already flat vectors
        pca_input = test_images
        reshape_shape = len(test_images), 784
    elif dataset.lower() == "cifar":
        if n_components is None:
            n_components = 128
        # For CIFAR, reshape from (N, 3, 32, 32) to (N, 3072)
        pca_input = test_images.reshape(len(test_images), -1)
        reshape_shape = len(test_images), 3, 32, 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'mnist' or 'cifar'.")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_reduced = pca.fit_transform(pca_input)
    
    pca_zeroed = np.zeros_like(pca_reduced)
    
    # Only keep the specified PC group
    pca_zeroed[:, keep_start_idx:keep_end_idx] = pca_reduced[:, keep_start_idx:keep_end_idx]
    
    # Reconstruct
    return pca.inverse_transform(pca_zeroed).reshape(reshape_shape)


def get_encoding_diff(model, original_img, noisy_img, dataset="mnist") -> np.ndarray:
    """
    Get the difference in bottleneck activation between original and noisy images.
    
    Args:
        model: The autoencoder model
        original_img: Original image
        noisy_img: Noisy image
        dataset: Dataset ('mnist' or 'cifar')
        
    Returns:
        Absolute difference in bottleneck activations
    """
    # Convert NumPy arrays to PyTorch tensors
    if isinstance(original_img, np.ndarray):
        original_img = torch.tensor(original_img, dtype=torch.float32)
    if isinstance(noisy_img, np.ndarray):
        noisy_img = torch.tensor(noisy_img, dtype=torch.float32)
    
    # Add batch dimension for CIFAR
    if dataset.lower() == "cifar":
        original_img = original_img.unsqueeze(0)
        noisy_img = noisy_img.unsqueeze(0)
    
    original_encoding = model.encode(original_img)
    noisy_encoding = model.encode(noisy_img)
    return np.abs((original_encoding - noisy_encoding).detach().numpy())


def evaluate_models(test_images, reconstructed_images, sae, dae, dataset="mnist"):
    """
    Evaluate models by measuring encoding differences between original and noisy images.
    
    Args:
        test_images: Original test images
        reconstructed_images: Noisy reconstructed images
        sae: SAE model
        dae: DAE model
        dataset: Dataset ('mnist' or 'cifar')
        
    Returns:
        Tuple of mean SAE and DAE differences
    """
    sae_diffs = []
    dae_diffs = []
    
    for i in range(len(test_images)):
        test_image = test_images[i]
        reconstructed_image = reconstructed_images[i]
        
        with torch.no_grad():
            sae_diff = get_encoding_diff(sae, test_image, reconstructed_image, dataset)
            dae_diff = get_encoding_diff(dae, test_image, reconstructed_image, dataset)

            sae_diffs.append(sae_diff)
            dae_diffs.append(dae_diff)
    
    return np.mean(np.vstack(sae_diffs), axis=0), np.mean(np.vstack(dae_diffs), axis=0)


def plot_neuron_comparison(results, manipulated_neurons, savepath, dataset="mnist", method="noise"):
    """
    Plot comparison of neuron responses to PC noise or zeroing.
    
    Args:
        results: Results dictionary
        manipulated_neurons: List of (start, end) PC ranges
        savepath: Path to save the plot
        dataset: Dataset ('mnist' or 'cifar')
        method: Method used ('noise' or 'zeroing')
    """
    # plt.rcParams.update({'font.size': 16})
    
    plt.figure(figsize=(12, 8))
    
    sae_colors = plt.cm.Blues(np.linspace(0.3, 1, len(manipulated_neurons)))
    dae_colors = plt.cm.Reds(np.linspace(0.3, 1, len(manipulated_neurons)))
    
    for i, neuron_pair in enumerate(manipulated_neurons):
        sae_data = np.vstack([run[0] for run in results[neuron_pair]])
        dae_data = np.vstack([run[1] for run in results[neuron_pair]])
        
        # Calculate statistics
        sae_mean = np.mean(sae_data, axis=0)
        sae_std = np.std(sae_data, axis=0)
        dae_mean = np.mean(dae_data, axis=0)
        dae_std = np.std(dae_data, axis=0)
        
        # Label depends on method
        if method == "noise":
            label_text = f"Noisy PCs: {manipulated_neurons[i][0]+1} - {manipulated_neurons[i][1]}"
        else:  # zeroing
            label_text = f"Kept PCs: {manipulated_neurons[i][0]+1} - {manipulated_neurons[i][1]}"
        
        # Plot SAE
        plt.plot(sae_mean, color=sae_colors[i], 
                label=f'SAE ({label_text})', 
                linewidth=2)
        plt.fill_between(range(len(sae_mean)), 
                        sae_mean - sae_std, 
                        sae_mean + sae_std,
                        color=sae_colors[i], alpha=0.1)
        
        # Plot DAE
        plt.plot(dae_mean, color=dae_colors[i], 
                label=f'DAE ({label_text})', 
                linewidth=2)
        plt.fill_between(range(len(dae_mean)),
                        dae_mean - dae_std,
                        dae_mean + dae_std,
                        color=dae_colors[i], alpha=0.1)
    
    title_method = "PC Noise" if method == "noise" else "PC Zeroing"
    plt.title(f"{dataset.upper()} {title_method}", fontsize=16, pad=20)
    plt.xlabel("Neuron Index", fontsize=16)
    plt.ylabel("Absolute Activation Difference", fontsize=16)
    
    legend = plt.legend(loc='upper right', 
                      fontsize=14,
                      framealpha=0.9,
                      edgecolor='black')
    
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def compute_pc_noise_analysis(num_models, manipulated_neurons, dataset="mnist", base_path="/home/david/"):
    """
    Compute PC noise analysis for all models.
    
    Args:
        num_models: Number of models to evaluate
        manipulated_neurons: List of (start, end) PC ranges
        dataset: Dataset ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    result_file = f"Results/{dataset}_pc_noise.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return np.load(result_file, allow_pickle=True).item()
    
    # Load test images based on dataset
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_tensor()
    else:
        test_images, _ = load_cifar_list()

    results = {pair: [] for pair in manipulated_neurons}

    for neuron_pair in tqdm(manipulated_neurons, desc="Processing PC ranges", leave=False):
        # Generate noisy images
        noisy_reconstructed = add_noise_and_reconstruct(
            test_images, 
            noise_scale=10,
            dataset=dataset, 
            start_noise_idx=neuron_pair[0], 
            end_noise_idx=neuron_pair[1]
        )
        
        for iteration in tqdm(range(num_models), desc=f"Testing models for PCs {neuron_pair}", leave=False):
            # Load models based on dataset
            if dataset.lower() == "mnist":
                model_path = f'{base_path}mnist_models/'
                sae = load_model(f"{model_path}sae/{iteration}", 'sae', 59)
                dae = load_model(f"{model_path}dae/{iteration}", 'dae', 59)
            else:
                model_path = f"{base_path}cifar_models/"
                sae = load_conv_model(f"{model_path}sae/{iteration}", 'sae', 59)
                dae = load_conv_model(f"{model_path}dae/{iteration}", 'sae', 59, size_ls=[128]*60)
            
            # Evaluate models
            sae_diffs, dae_diffs = evaluate_models(test_images, noisy_reconstructed, sae, dae, dataset)
            results[neuron_pair].append((sae_diffs, dae_diffs))
    
    np.save(result_file, results)
    return results


def compute_pc_zeroing_analysis(num_models, manipulated_neurons, dataset="mnist", base_path="/home/david/"):
    """
    Compute PC zeroing analysis for all models (keeping only specified PC groups).
    
    Args:
        num_models: Number of models to evaluate
        manipulated_neurons: List of (start, end) PC ranges
        dataset: Dataset ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    result_file = f"Results/{dataset}_pc_zeroing.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return np.load(result_file, allow_pickle=True).item()
    
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_tensor()
    else:
        test_images, _ = load_cifar_list()

    results = {pair: [] for pair in manipulated_neurons}

    for neuron_pair in tqdm(manipulated_neurons, desc="Processing PC ranges", leave=False):
        zeroed_reconstructed = add_zeroing_and_reconstruct(
            test_images,
            dataset=dataset, 
            keep_start_idx=neuron_pair[0], 
            keep_end_idx=neuron_pair[1]
        )
        
        for iteration in tqdm(range(num_models), desc=f"Testing models for PCs {neuron_pair}", leave=False):
            # Load models based on dataset
            if dataset.lower() == "mnist":
                model_path = f'{base_path}mnist_models/'
                sae = load_model(f"{model_path}sae/{iteration}", 'sae', 59)
                dae = load_model(f"{model_path}dae/{iteration}", 'dae', 59)
            else:
                model_path = f"{base_path}cifar_models/"
                sae = load_conv_model(f"{model_path}sae/{iteration}", 'sae', 59)
                dae = load_conv_model(f"{model_path}dae/{iteration}", 'sae', 59, size_ls=[128]*60)
            
            # Evaluate models
            sae_diffs, dae_diffs = evaluate_models(test_images, zeroed_reconstructed, sae, dae, dataset)
            results[neuron_pair].append((sae_diffs, dae_diffs))
    
    np.save(result_file, results)
    return results


def create_ranking_heatmaps(results, manipulated_neurons, dataset="mnist", method="noise"):
    """
    Create improved heatmaps showing rankings of PC impact on neurons.
    
    Args:
        results: Results dictionary
        manipulated_neurons: List of (start, end) PC ranges
        dataset: Dataset ('mnist' or 'cifar')
        method: Method used ('noise' or 'zeroing')
        
    Returns:
        Tuple of SAE and DAE rankings
    """
    sample_data = next(iter(results.values()))[0]
    num_neurons = len(sample_data[0])
    
    # Initialize arrays to store mean activation differences
    sae_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons))
    dae_activation_matrix = np.zeros((len(manipulated_neurons), num_neurons))
    
    # Calculate mean activation differences for each PC range and each neuron
    for i, pc_range in enumerate(manipulated_neurons):
        # Get all runs for this PC range
        runs = results[pc_range]
        
        # Stack all SAE and DAE differences for this PC range
        sae_diffs = np.vstack([run[0] for run in runs])
        dae_diffs = np.vstack([run[1] for run in runs])
        
        # Calculate mean across runs
        sae_mean = np.mean(sae_diffs, axis=0)
        dae_mean = np.mean(dae_diffs, axis=0)
        
        # Store in matrices
        sae_activation_matrix[i, :] = sae_mean
        dae_activation_matrix[i, :] = dae_mean
    
    # Calculate rankings for each neuron (1 = highest activation difference)
    sae_rankings = np.zeros_like(sae_activation_matrix, dtype=int)
    dae_rankings = np.zeros_like(dae_activation_matrix, dtype=int)
    
    for neuron in range(num_neurons):
        # Get activation differences for this neuron across all PC ranges
        sae_neuron_diffs = sae_activation_matrix[:, neuron]
        dae_neuron_diffs = dae_activation_matrix[:, neuron]
        
        if method != "noise":
            # For zeros, higher activation difference is worse, so invert ranking
            sae_neuron_diffs *= -1
            dae_neuron_diffs *= -1
        # Calculate rankings using argsort and flipping so that 1 = highest activation difference
        sae_rankings[:, neuron] = np.argsort(np.argsort(-sae_neuron_diffs)) + 1
        dae_rankings[:, neuron] = np.argsort(np.argsort(-dae_neuron_diffs)) + 1
    
    pc_labels = [f"{r[0]+1}-{r[1]}" for r in manipulated_neurons]
    
    # plt.rcParams.update({'font.size': 14})
    
    num_ranks = len(manipulated_neurons)
    blues = plt.cm.Blues_r(np.linspace(0, 1, num_ranks + 1))
    discrete_blues = ListedColormap(blues)
    reds = plt.cm.Reds_r(np.linspace(0, 1, num_ranks + 1))
    discrete_reds = ListedColormap(reds)
    
    bounds = [i + 0.5 for i in range(num_ranks + 1)]
    norm = BoundaryNorm(bounds, num_ranks)
    
    neuron_group_start_indices = [pair[0]+1 for pair in manipulated_neurons]

    display_indices = [idx for idx in neuron_group_start_indices if idx != 7 and idx != 17]
    
    fig = plt.figure(figsize=(6.266, 2), dpi=300)
    gs = gridspec.GridSpec(1, 2, wspace=0.2)

    fig.subplots_adjust(bottom=0.22)
    
    # SAE Heatmap
    ax1 = plt.subplot(gs[0])
    
    sns.heatmap(sae_rankings, annot=False, cmap=discrete_blues, 
                cbar=True, square=False, linewidths=0,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=pc_labels, ax=ax1)
    
    # Add gray lines at neuron group boundaries
    for i, start_idx in enumerate(neuron_group_start_indices):
        if i > 0:
            ax1.axvline(x=start_idx-1, color='gray', linewidth=2)
    
    ax1.set_xticks([idx-1+0.5 for idx in display_indices])
    ax1.set_xticklabels(display_indices)
    
    # Colorbar
    cbar1 = ax1.collections[0].colorbar
    cbar1.set_ticks(list(range(1, num_ranks + 1)))
    cbar1.set_ticklabels(list(range(1, num_ranks + 1)))
    cbar1.minorticks_off()
    cbar1.ax.invert_yaxis()
    cbar1.outline.set_linewidth(0.7)
    cbar1.outline.set_edgecolor('black')
    
    ax1.set_title("AE", fontsize=11)
    ax1.set_xlabel("Neuron Index")
    ax1.set_ylabel("PC Range")
    
    # DAE Heatmap
    ax2 = plt.subplot(gs[1])
    
    sns.heatmap(dae_rankings, annot=False, cmap=discrete_reds, 
                cbar=True, square=False, linewidths=0,
                norm=norm,
                xticklabels=range(1, num_neurons+1),
                yticklabels=[], ax=ax2)
    
    # Add gray lines at neuron group boundaries
    for i, start_idx in enumerate(neuron_group_start_indices):
        if i > 0:
            ax2.axvline(x=start_idx-1, color='gray', linewidth=2.0)
    
    ax2.set_xticks([idx-1+0.5 for idx in display_indices])
    ax2.set_xticklabels(display_indices)
    
    # Colorbar
    cbar2 = ax2.collections[0].colorbar
    cbar2.set_ticks(list(range(1, num_ranks + 1)))
    cbar2.set_ticklabels(list(range(1, num_ranks + 1)))
    cbar2.set_label('Ranking')
    cbar2.minorticks_off()
    cbar2.ax.invert_yaxis()
    cbar2.outline.set_linewidth(0.7)
    cbar2.outline.set_edgecolor('black')
    
    ax2.set_title("Dev-AE", fontsize=11)
    ax2.set_xlabel("Neuron Index")

    # fig.suptitle("PC Noise Impact Rankings", fontsize=18, y=1.05)
    
    # plt.tight_layout()
    
    fig.savefig(f"Results/figures/png/{dataset}_pc_{method}_combined_rankings.png", dpi=300)
    fig.savefig(f"Results/figures/svg/{dataset}_pc_{method}_combined_rankings.svg")
    plt.savefig(f"Results/figures/pdf/{dataset}_pc_{method}_combined_rankings.pdf")
    plt.savefig(f"Results/figures/eps/{dataset}_pc_{method}_combined_rankings.eps")
    plt.close()
    
    return sae_rankings, dae_rankings


def analyze_and_visualize_pc_noise(dataset, manipulated_neurons):
    """
    Analyze and visualize PC noise results.
    
    Args:
        dataset: Dataset ('mnist' or 'cifar')
        
    Returns:
        Tuple of SAE and DAE rankings
    """
    # Load the results
    results_file = f"Results/{dataset}_pc_noise.npy"
    results = np.load(results_file, allow_pickle=True).item()

    sae_rankings, dae_rankings = create_ranking_heatmaps(results, manipulated_neurons, dataset, method="noise")
    
    return sae_rankings, dae_rankings


def analyze_and_visualize_pc_zeroing(dataset, manipulated_neurons):
    """
    Analyze and visualize PC zeroing results.
    
    Args:
        dataset: Dataset ('mnist' or 'cifar')
        
    Returns:
        Tuple of SAE and DAE rankings
    """
    # Load the results
    results_file = f"Results/{dataset}_pc_zeroing.npy"
    results = np.load(results_file, allow_pickle=True).item()

    sae_rankings, dae_rankings = create_ranking_heatmaps(results, manipulated_neurons, dataset, method="zeroing")
    
    return sae_rankings, dae_rankings


def run_pc_noise_analysis(num_models, dataset, base_path, manipulated_neurons):
    """
    Run the complete PC noise analysis.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    compute_pc_noise_analysis(num_models, manipulated_neurons, dataset, base_path)
    results = np.load(f"Results/{dataset}_pc_noise.npy", allow_pickle=True).item()
    plot_neuron_comparison(results, manipulated_neurons, f"Results/figures/png/{dataset}_pc_noise.png", dataset, method="noise")
    analyze_and_visualize_pc_noise(dataset, manipulated_neurons)


def run_pc_zeroing_analysis(num_models, dataset, base_path, manipulated_neurons):
    """
    Run the complete PC zeroing analysis.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    compute_pc_zeroing_analysis(num_models, manipulated_neurons, dataset, base_path)
    results = np.load(f"Results/{dataset}_pc_zeroing.npy", allow_pickle=True).item()
    plot_neuron_comparison(results, manipulated_neurons, f"Results/figures/png/{dataset}_pc_zeroing.png", dataset, method="zeroing")
    analyze_and_visualize_pc_zeroing(dataset, manipulated_neurons)


def run_all_pc_analyses(num_models, dataset, base_path, manipulated_neurons):
    """
    Run both noise and zeroing PC analyses.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    print(f"Running PC noise analysis for {dataset}...")
    run_pc_noise_analysis(num_models, dataset, base_path, manipulated_neurons)
    
    print(f"Running PC zeroing analysis for {dataset}...")
    run_pc_zeroing_analysis(num_models, dataset, base_path, manipulated_neurons)

if __name__ == "__main__":
    run_all_pc_analyses(40, "mnist", '/home/david/', manipulated_neurons=[(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)])
    run_all_pc_analyses(1, "cifar", '/home/david/', manipulated_neurons=[(0, 6), (6, 10), (10, 16), (16, 28), (28, 48), (48, 90), (90, 128)])