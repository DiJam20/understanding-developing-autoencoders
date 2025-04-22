import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
from torch.nn.functional import mse_loss

from autoencoder import *
from model_utils import *
from solver import *

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11,
    'axes.titlesize': 11
})


def add_noise_and_calculate_loss(model, image, start_idx, end_idx, dataset, normalize=False):
    """
    Add noise to a specific section of the encoding and calculate reconstruction loss.
    
    Args:
        model: The autoencoder model to evaluate
        image: The input image to evaluate
        start_idx: The start index of the section to add noise
        end_idx: The end index of the section to add noise
        dataset: The dataset being used ('mnist' or 'cifar')
        normalize: Whether to normalize the noise by the number of neurons
        
    Returns:
        Tuple of loss, encoded representation, and decoded image
    """
    with torch.no_grad():
        if dataset == 'cifar':
            original_image = image.clone()

            # Reshape flattened CIFAR image: [3072] to [1, 3, 32, 32]
            image = image.reshape(1, 3, 32, 32)

            encoded = model.encode(image).squeeze(0)
            
            mean, std = encoded[start_idx:end_idx].mean(), encoded[start_idx:end_idx].std()
            
            # Add noise to the specified section of the encoding
            # If normalize is True, add noise normalized to the number of neurons
            if normalize:
                normalized_std = (1 / (end_idx - start_idx))
                encoded[start_idx:end_idx] = encoded[start_idx:end_idx] + torch.randn(end_idx - start_idx) * normalized_std + mean
            else:
                encoded[start_idx:end_idx] = torch.randn(end_idx - start_idx) * std * 2 + mean
            
            # Apply linear transformation to encoded representation
            encoded_linear = model.linear(encoded.unsqueeze(0))
            # Reshape to match decoder input shape (1, 64, 4, 4)
            encoded_linear = encoded_linear.reshape(1, -1, 4, 4)
            
            decoded = model.decoder(encoded_linear)
            decoded_flat = decoded.squeeze(0).flatten()
            loss = mse_loss(decoded_flat, original_image).item()
            
            return loss, encoded, decoded
        else:
            encoded = model.encode(image)
            mean, std = encoded[start_idx:end_idx].mean(), encoded[start_idx:end_idx].std()
            encoded[start_idx:end_idx] = torch.randn(end_idx - start_idx) * std * 2 + mean
            decoded = model.decode(encoded)
            loss = mse_loss(decoded, image).item()
    return loss, encoded, decoded


def evaluate_single_model(model, test_images, neuron_groups, dataset):
    """
    Evaluate a model by adding noise to different groups of neurons.
    
    Args:
        model: The autoencoder model to evaluate
        test_images: Dataset of test images
        neuron_groups: List of indices defining the end of each group
    
    Returns:
        List of average losses for each group
    """
    num_groups = len(neuron_groups)
    group_losses = [0.0] * num_groups
    
    start_indices = [0] + [neuron_groups[i-1] for i in range(1, num_groups)]
    
    for image in test_images:
        for i in range(num_groups):
            start_idx = start_indices[i]
            end_idx = neuron_groups[i]
            loss, _, _ = add_noise_and_calculate_loss(model, image, start_idx, end_idx, dataset, normalize=True)
            group_losses[i] += loss
    
    # Average over all test images
    for loss in group_losses:
        loss /= len(test_images)

    # Normalize by number of groups
    # for i in range(num_groups):
    #     group_losses[i] /= (neuron_groups[i] - start_indices[i])
    
    return group_losses


def evaluate_models_with_averaging(num_models, test_images, neuron_groups, dataset, base_path):
    """
    Evaluate multiple models and average the results.
    
    Args:
        test_images: Dataset of test images
        neuron_groups: List of indices defining the end of each group
        num_models: Number of models to evaluate and average
        
    Returns:
        Tuple of activation difference lists
    """

    result_file = f"Results/encoding_noise_{dataset}.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    sae_results = []
    dae_results = []

    for iteration in tqdm(range(num_models), desc="Processing models", leave=False):
        modelpath = f'{base_path}{dataset}_models/'
        # epoch = len(np.load(f'{modelpath}dae/size_each_epoch.npy')) - 1
        epoch = 59
        if dataset == 'mnist':
            sae = load_model(modelpath+'sae/'+str(iteration), 'sae', epoch)
            dae = load_model(modelpath+'dae/'+str(iteration), 'dae', 59)
        else:
            sae = load_conv_model(modelpath+'sae/'+str(iteration), 'sae', epoch)
            dae = load_conv_model(modelpath+'dae/'+str(iteration), 'dae', 59, [128] * (epoch + 1))
        
        sae_result = evaluate_single_model(sae, test_images, neuron_groups, dataset)
        dae_result = evaluate_single_model(dae, test_images, neuron_groups, dataset)
        
        sae_results.append(sae_result)
        dae_results.append(dae_result)

    np.save(result_file, (sae_results, dae_results))
    
    return None


def plot_heatmap(sae_results, dae_results, neuron_groups, dataset):
    """
    Create a heatmap visualization of loss values across neuron groups.
    
    Args:
        sae_results: List of lists of losses for SAE models
        dae_results: List of lists of losses for DAE models
        neuron_groups: List of indices defining the end of each group
    """
    sae_means = np.mean(sae_results, axis=0)
    dae_means = np.mean(dae_results, axis=0)
    
    blues = plt.cm.Blues(np.linspace(0.2, 0.8, 256))
    blues = ListedColormap(blues)
    
    reds = plt.cm.Reds(np.linspace(0.2, 0.8, 256))
    reds = ListedColormap(reds)
    
    # Calculate start indices for each neuron group
    start_indices = [1]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1] + 1)

    # Create labels for each neuron group
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start}-{end}")
    
    # Min and max of both models for fair comparison
    vmin = min(np.min(sae_means), np.min(dae_means))
    vmax = max(np.max(sae_means), np.max(dae_means))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    
    # SAE heatmap
    sns.heatmap(sae_means.reshape(1, -1), annot=False, cmap=blues,
               xticklabels=x_labels, yticklabels=["AE"], ax=ax1, 
               vmin=vmin, vmax=vmax,
               cbar_kws={"label": "Reconstruction Loss"})
    ax1.set_title('AE Reconstruction Loss')
    ax1.set_xlabel('')
    
    # DAE heatmap
    sns.heatmap(dae_means.reshape(1, -1), annot=False, cmap=reds,
               xticklabels=x_labels, yticklabels=["DevAE"], ax=ax2, 
               vmin=vmin, vmax=vmax,
               cbar_kws={"label": "Reconstruction Loss"})
    ax2.set_title('DevAE Reconstruction Loss')
    ax2.set_xlabel('Manipulated Neuron Groups')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f'Results/encoding_noise_heatmap_{dataset}.png', dpi=300, bbox_inches='tight')
    plt.close

    return None


def collect_noisy_images(image, neuron_groups, dataset, base_path):
    """
    Collect noisy images and their loss values for both SAE and DAE models.
    
    Args:
        image: The input image to evaluate
        neuron_groups: List of indices defining the end of each group
        dataset: The dataset being used ('mnist' or 'cifar')
        
    Returns:
        Tuple of lists of noisy images and loss values for SAE and DAE models
    """
    num_groups = len(neuron_groups)
    start_indices = [0] + [neuron_groups[i-1] for i in range(1, num_groups)]
    group_sizes = [neuron_groups[i] - start_indices[i] for i in range(num_groups)]

    modelpath = f'{base_path}{dataset}_models'
    if dataset == 'mnist':
        sae = load_model(f'{modelpath}/sae/0', 'sae', 59)
        dae = load_model(f'{modelpath}/dae/0', 'dae', 59)
    else:
        sae = load_conv_model(f'{modelpath}/sae/0', 'sae', 59)
        dae = load_conv_model(f'{modelpath}/dae/3', 'dae', 59, [128] * 60)

    sae_noisy_images = []
    dae_noisy_images = []
    sae_losses = []
    dae_losses = []

    for _, (start_idx, group_size) in enumerate(zip(start_indices, group_sizes)):
        end_idx = start_idx + group_size
        sae_loss, _, sae_noisy = add_noise_and_calculate_loss(sae, image.clone(), start_idx, end_idx, dataset, normalize=True)
        dae_loss, _, dae_noisy = add_noise_and_calculate_loss(dae, image.clone(), start_idx, end_idx, dataset, normalize=True)

        sae_noisy_images.append(sae_noisy)
        dae_noisy_images.append(dae_noisy)
        sae_losses.append(sae_loss)
        dae_losses.append(dae_loss)

    return sae_noisy_images, dae_noisy_images, sae_losses, dae_losses


def plot_noisy_images(sae_noisy_images, dae_noisy_images, sae_losses, dae_losses, neuron_groups, dataset):
    """
    Create a grid of reconstructed images showing the effect of noise on different neuron groups.
    
    Args:
        sae_noisy_images: List of noisy reconstructed images from SAE model
        dae_noisy_images: List of noisy reconstructed images from DAE model
        sae_losses: Loss values for each SAE image
        dae_losses: Loss values for each DAE image
        neuron_groups: List of indices defining the end of each group
        dataset: The dataset being used ('mnist' or 'cifar')
    """
    num_groups = len(neuron_groups)
    
    width = 6.266
    if dataset == 'mnist':
        width = 6.266 * 0.8
    fig, axes = plt.subplots(2, num_groups, figsize=(width, 2))
    
    # Calculate start indices for each neuron group
    start_indices = [0] + [neuron_groups[i-1] for i in range(1, num_groups)]
    
    # Create labels for each neuron group
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Process each neuron group for both models
    for group_idx in range(num_groups):
        # SAE Image (top row)
        sae_loss = sae_losses[group_idx]
        sae_decoded = sae_noisy_images[group_idx]
        
        if dataset == 'cifar':
            sae_img = sae_decoded.squeeze(0).permute(1, 2, 0).detach().numpy()
            # Normalize from [-1,1] to [0,1]
            sae_img = (sae_img + 1) / 2
            axes[0, group_idx].imshow(sae_img)
        else:
            sae_img = sae_decoded.reshape(28, 28).detach().numpy()
            axes[0, group_idx].imshow(sae_img, cmap='gray', vmin=-1, vmax=3)
        
        # DAE Image (bottom row)
        dae_loss = dae_losses[group_idx]
        dae_decoded = dae_noisy_images[group_idx]
        
        if dataset == 'cifar':
            dae_img = dae_decoded.squeeze(0).permute(1, 2, 0).detach().numpy()
            # Normalize from [-1,1] to [0,1]
            dae_img = (dae_img + 1) / 2
            axes[1, group_idx].imshow(dae_img)
        else:
            dae_img = dae_decoded.reshape(28, 28).detach().numpy()
            axes[1, group_idx].imshow(dae_img, cmap='gray', vmin=-1, vmax=3)
                
        # Add loss labels
        axes[0, group_idx].text(1, -0.03, f"{sae_loss:.2f}", 
                               size=11, ha="right", va="bottom", 
                               color='black', 
                               bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7),
                               transform=axes[0, group_idx].transAxes)
        axes[1, group_idx].text(1, -0.03, f"{dae_loss:.2f}", 
                               size=11, ha="right", va="bottom", 
                               color='black', 
                               bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7),
                               transform=axes[1, group_idx].transAxes)
        
        axes[0, group_idx].set_xticks([])
        axes[0, group_idx].set_yticks([])
        axes[1, group_idx].set_xticks([])
        axes[1, group_idx].set_yticks([])

        for edge in ['top', 'right', 'bottom', 'left']:
            axes[0, group_idx].spines[edge].set_visible(False)
            axes[1, group_idx].spines[edge].set_visible(False)
        
        if group_idx == 0:
            axes[0, group_idx].set_ylabel("AE", fontsize=11, rotation=0, labelpad=20, va='center')
            axes[1, group_idx].set_ylabel("Dev-AE", fontsize=11, rotation=0, labelpad=20, va='center')
        
        axes[0, group_idx].set_title(x_labels[group_idx], fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f'Results/figures/png/noisy_images_{dataset}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Results/figures/svg/noisy_images_{dataset}.svg', bbox_inches='tight')
    plt.savefig(f'Results/figures/pdf/noisy_images_{dataset}.pdf', bbox_inches='tight')
    plt.close()


def plot_bar_comparison(sae_results, dae_results, neuron_groups, dataset):
    """
    Create a bar plot comparing SAE and DAE reconstruction loss across neuron groups.
    
    Args:
        sae_results: List of lists of losses for SAE models
        dae_results: List of lists of losses for DAE models
        neuron_groups: List of indices defining the end of each group
        dataset: The dataset being used ('mnist' or 'cifar')
    """
    # Calculate means and standard deviations
    sae_means = np.mean(sae_results, axis=0)
    dae_means = np.mean(dae_results, axis=0)
    sae_stds = np.std(sae_results, axis=0)
    dae_stds = np.std(dae_results, axis=0)
    
    # Calculate start indices for each neuron group
    start_indices = [1]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1] + 1)

    # Create labels for each neuron group
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start}-{end}")
    
    # plt.rc('font', size=20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.268*0.49, 2.6), dpi=300)
    
    # Plot SAE
    x_indices = np.arange(len(neuron_groups))
    sae_bars = ax1.bar(x_indices, sae_means, color='#1a7adb', yerr=sae_stds, capsize=4)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=90)
    if dataset == 'mnist':
        ax1.set_ylabel('Reconstruction Loss ($\\times10^{4}$)', fontsize=11)
        ax1.set_yticks([10000, 20000, 30000], [1, 2, 3])
    if dataset == 'cifar':
        ax1.set_ylabel('Reconstruction Loss ($\\times10^{3}$)', fontsize=11)
        ax1.set_yticks([4000, 8000], [4, 8])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot DAE
    dae_bars = ax2.bar(x_indices, dae_means, color='#e82817', yerr=dae_stds, capsize=4)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, rotation=90)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')

    ax1.legend([sae_bars, dae_bars], ['AE', 'Dev-AE'], loc='upper left', bbox_to_anchor=(0, 1.2))

    
    # Set the same y-limit for both plots
    max_val = max(
        max(sae_means) + max(sae_stds),
        max(dae_means) + max(dae_stds)
    )
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    # fig.suptitle(f'Reconstruction Loss Across Neuron Groups ({dataset.upper()})', fontsize=24)
    fig.supxlabel('Neuron Group', x=0.57, y=0.1, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.1)

    plt.savefig(f'Results/figures/png/bar_comparison_{dataset}.png', dpi=300)
    plt.savefig(f'Results/figures/pdf/bar_comparison_{dataset}.pdf')
    plt.close()


def run_encoding_noise_analysis(num_models, size_ls, dataset='mnist', base_path='/home/david/'):
    """
    Run the encoding noise analysis for SAE and DAE models.
    """
    if dataset == 'mnist':
        test_images, _ = load_mnist_tensor()
    else:
        test_images, _ = load_cifar_tensor()

    neuron_groups = sorted(set(size_ls))
    
    evaluate_models_with_averaging(num_models, test_images, neuron_groups, dataset, base_path)

    sae_results, dae_results = np.load(f"Results/encoding_noise_{dataset}.npy")
        
    # Plot heatmap and bar comparison
    plot_heatmap(sae_results, dae_results, neuron_groups, dataset)
    plot_bar_comparison(sae_results, dae_results, neuron_groups, dataset)
    
    # Collect and plot example noisy images
    if dataset == 'mnist':
        test_image = test_images[0]
    else:
        test_image = test_images[8]
    
    sae_noisy_images, dae_noisy_images, sae_losses, dae_losses = collect_noisy_images(test_image, neuron_groups, dataset, base_path)
    plot_noisy_images(sae_noisy_images, dae_noisy_images, sae_losses, dae_losses, neuron_groups, dataset)

if __name__ == "__main__":
    mnist_size_ls = [4, 4, 4, 4, 4, 10,
        10, 10, 10, 10, 16, 16,
        16, 16, 16, 16, 16, 24,
        24, 24, 24, 24, 24, 24, 
        32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 
        32, 32, 32, 32, 32, 32, 
        32, 32, 32, 32, 32, 32, 
        32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32,]
    
    cifar_size_ls = [6,   6,   6,   6,   6,   6,    # 6
					10,  10,  10,  10,  10,  10,    # 6
					16,  16,  16,  16,  16,  16,    # 6
					28,  28,  28,  28,  28,  28,    # 6
					48,  48,  48,  48,  48,  48,  48,  48, 48, # 9
					90,  90,  90,  90,  90,  90,  90,  90,  90,  90, #10
					128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
					128, 128, 128, 128, 128, 128, 128 # 17
					]

    run_encoding_noise_analysis(40, mnist_size_ls, dataset='mnist', base_path='/home/david/')
    run_encoding_noise_analysis(10, cifar_size_ls, dataset='cifar', base_path='/home/david/')