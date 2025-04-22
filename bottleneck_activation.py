import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from autoencoder import *
from model_utils import *
from solver import *

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11
})


def get_bottleneck_activation(model, img: torch.Tensor) -> np.ndarray:
    """
    Get the bottleneck activation of a model for a given image.
    
    Args:
        model: The autoencoder model
        img: The input image
        
    Returns:
        The bottleneck activation
    """
    if isinstance(model, ConvAutoencoder):
        original_encoding = model.encode(img.reshape(1, 3, 32, 32))
    else:
        original_encoding = model.encode(img)
    return original_encoding.detach().numpy()    


def evaluate_models(test_images: np.ndarray, sae, dae) -> tuple:
    """
    Evaluate the models by calculating the percentage of zeros for each neuron
    and the mean of non-zero activations for each neuron.

    Args:
        test_images: The test images
        sae: The SAE model
        dae: The DAE model

    Returns:
        Tuple containing the percentage of zeros for each neuron and the mean of non-zero activations for each neuron
    """
    # Per neuron statistics for zeros tracking
    sae_zeros = None
    dae_zeros = None
    
    # For tracking non-zero activations
    sae_nonzero_counts = None
    dae_nonzero_counts = None
    sae_nonzero_sum = None
    dae_nonzero_sum = None

    # Per image statistics
    sae_per_image_zeros = []
    dae_per_image_zeros = []
    sae_per_image_nonzero_mean = []
    dae_per_image_nonzero_mean = []

    num_images = 0
    if isinstance(sae, NonLinearAutoencoder):
        threshold = 0
    else:
        threshold = 1e-4

    for i in range(len(test_images)):
        test_image = test_images[i]
        num_images += 1
        
        with torch.no_grad():
            sae_activations = get_bottleneck_activation(sae, test_image)
            dae_activations = get_bottleneck_activation(dae, test_image)

            if sae_zeros is None:
                sae_zeros = np.zeros_like(sae_activations)
                dae_zeros = np.zeros_like(dae_activations)
                sae_nonzero_counts = np.zeros_like(sae_activations)
                dae_nonzero_counts = np.zeros_like(dae_activations)
                sae_nonzero_sum = np.zeros_like(sae_activations)
                dae_nonzero_sum = np.zeros_like(dae_activations)
            
            # Count zeros
            sae_is_zero = (np.abs(sae_activations) <= threshold)
            dae_is_zero = (np.abs(dae_activations) <= threshold)
            sae_zeros += sae_is_zero
            dae_zeros += dae_is_zero
            
            # Count and sum non-zero activations
            sae_nonzero_counts += ~sae_is_zero
            dae_nonzero_counts += ~dae_is_zero
            sae_nonzero_sum += np.where(sae_is_zero, 0, np.abs(sae_activations))
            dae_nonzero_sum += np.where(dae_is_zero, 0, np.abs(dae_activations))

            # Percentage of zeros for this image
            sae_per_image_zeros.append(np.mean(sae_is_zero) * 100)
            dae_per_image_zeros.append(np.mean(dae_is_zero) * 100)
            
            flat_sae_activations = np.abs(sae_activations).reshape(-1)
            flat_dae_activations = np.abs(dae_activations).reshape(-1)
            flat_sae_is_zero = sae_is_zero.reshape(-1)
            flat_dae_is_zero = dae_is_zero.reshape(-1)
            
            # Mean of non-zero activations for this image
            nonzero_sae_values = flat_sae_activations[~flat_sae_is_zero] if flat_sae_is_zero.any() else flat_sae_activations
            nonzero_dae_values = flat_dae_activations[~flat_dae_is_zero] if flat_dae_is_zero.any() else flat_dae_activations
            
            sae_per_image_nonzero_mean.append(np.mean(nonzero_sae_values) if len(nonzero_sae_values) > 0 else 0)
            dae_per_image_nonzero_mean.append(np.mean(nonzero_dae_values) if len(nonzero_dae_values) > 0 else 0)

    # Per neuron statistics
    sae_zeros_percent = (sae_zeros / num_images) * 100
    dae_zeros_percent = (dae_zeros / num_images) * 100
    
    # Mean of non-zero activations for each neuron
    sae_nonzero_mean = np.divide(sae_nonzero_sum, sae_nonzero_counts, 
                                out=np.zeros_like(sae_nonzero_sum), 
                                where=sae_nonzero_counts > 0)
    dae_nonzero_mean = np.divide(dae_nonzero_sum, dae_nonzero_counts, 
                                out=np.zeros_like(dae_nonzero_sum), 
                                where=dae_nonzero_counts > 0)

    sae_per_image_zeros = np.array(sae_per_image_zeros)
    dae_per_image_zeros = np.array(dae_per_image_zeros)
    sae_per_image_nonzero_mean = np.array(sae_per_image_nonzero_mean)
    dae_per_image_nonzero_mean = np.array(dae_per_image_nonzero_mean)
    
    return (sae_zeros_percent, dae_zeros_percent,
            sae_per_image_zeros, dae_per_image_zeros,
            sae_nonzero_mean, dae_nonzero_mean,
            sae_per_image_nonzero_mean, dae_per_image_nonzero_mean)


def compute_bottleneck_activation(num_models: int, dataset: str, base_path: str):
    """
    Compute the bottleneck activation for all models and save the results to a file.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    if dataset.lower() == "mnist":
        model_path = f'{base_path}mnist_models/'
    elif dataset.lower() == "cifar":
        model_path = f"{base_path}cifar_models/"
    
    result_file = f"Results/{dataset}_bottleneck_activation.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_tensor()
    else:
        test_images, _ = load_cifar_tensor()

    # Per neuron statistics
    all_sae_zeros = []
    all_dae_zeros = []
    all_sae_nonzero_means = []
    all_dae_nonzero_means = []

    # Per image statistics
    all_sae_per_image_zeros = []
    all_dae_per_image_zeros = []
    all_sae_per_image_nonzero_means = []
    all_dae_per_image_nonzero_means = []

    for iteration in tqdm(range(num_models), desc=f"Evaluating all models", leave=True):
        # Load models based on dataset
        if dataset.lower() == "mnist":
            sae = load_model(model_path+'sae/'+str(iteration), 'sae', 59)
            dae = load_model(model_path+'dae/'+str(iteration), 'dae', 59)
        else:
            sae = load_conv_model(model_path+'sae/'+str(iteration), 'sae', 59)
            dae = load_conv_model(model_path+'dae/'+str(iteration), 'dae', 59, size_ls=[128]*60)
        
        (sae_zeros, dae_zeros,
         sae_per_image_zeros, dae_per_image_zeros,
         sae_nonzero_mean, dae_nonzero_mean,
         sae_per_image_nonzero_mean, dae_per_image_nonzero_mean) = evaluate_models(test_images, sae, dae)
        
        all_sae_zeros.append(sae_zeros)
        all_dae_zeros.append(dae_zeros)
        all_sae_nonzero_means.append(sae_nonzero_mean)
        all_dae_nonzero_means.append(dae_nonzero_mean)

        all_sae_per_image_zeros.append(sae_per_image_zeros)
        all_dae_per_image_zeros.append(dae_per_image_zeros)
        all_sae_per_image_nonzero_means.append(sae_per_image_nonzero_mean)
        all_dae_per_image_nonzero_means.append(dae_per_image_nonzero_mean)

    # Average per neuron statistics
    sae_zeros_sum = np.zeros_like(all_sae_zeros[0])
    dae_zeros_sum = np.zeros_like(all_dae_zeros[0])
    sae_nonzero_means_sum = np.zeros_like(all_sae_nonzero_means[0])
    dae_nonzero_means_sum = np.zeros_like(all_dae_nonzero_means[0])
    
    for i in range(num_models):
        sae_zeros_sum += all_sae_zeros[i]
        dae_zeros_sum += all_dae_zeros[i]
        sae_nonzero_means_sum += all_sae_nonzero_means[i]
        dae_nonzero_means_sum += all_dae_nonzero_means[i]
    
    mean_sae_zeros = sae_zeros_sum / num_models
    mean_dae_zeros = dae_zeros_sum / num_models
    mean_sae_nonzero = sae_nonzero_means_sum / num_models
    mean_dae_nonzero = dae_nonzero_means_sum / num_models
    
    # Average per image statistics
    sae_per_image_zeros_avg = np.zeros_like(all_sae_per_image_zeros[0])
    dae_per_image_zeros_avg = np.zeros_like(all_dae_per_image_zeros[0])
    sae_per_image_nonzero_means_avg = np.zeros_like(all_sae_per_image_nonzero_means[0])
    dae_per_image_nonzero_means_avg = np.zeros_like(all_dae_per_image_nonzero_means[0])
    
    for i in range(num_models):
        sae_per_image_zeros_avg += all_sae_per_image_zeros[i]
        dae_per_image_zeros_avg += all_dae_per_image_zeros[i]
        sae_per_image_nonzero_means_avg += all_sae_per_image_nonzero_means[i]
        dae_per_image_nonzero_means_avg += all_dae_per_image_nonzero_means[i]
    
    sae_per_image_zeros_avg /= num_models
    dae_per_image_zeros_avg /= num_models
    sae_per_image_nonzero_means_avg /= num_models
    dae_per_image_nonzero_means_avg /= num_models
    
    # Create Results directory if it doesn't exist
    os.makedirs("Results", exist_ok=True)
    
    np.save(result_file, {
        # Save only the metrics we care about
        'mean_sae_zeros': mean_sae_zeros,
        'mean_dae_zeros': mean_dae_zeros,
        'mean_sae_nonzero': mean_sae_nonzero,
        'mean_dae_nonzero': mean_dae_nonzero,
        
        'sae_per_image_zeros': sae_per_image_zeros_avg,
        'dae_per_image_zeros': dae_per_image_zeros_avg,
        'sae_per_image_nonzero_means': sae_per_image_nonzero_means_avg,
        'dae_per_image_nonzero_means': dae_per_image_nonzero_means_avg
    })


def plot_activation_per_neuron(dataset: str):
    """
    Plot the mean non-zero activation per neuron for the SAE and DAE models,
    grouped by neuron ranges and displayed as a bar chart with error bars.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    if dataset.lower() == "mnist":
        neuron_groups = [4, 10, 16, 24, 32]
    elif dataset.lower() == "cifar":
        neuron_groups = [6, 10, 16, 28, 48, 90, 128]
        
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    
    sae_mean = results['mean_sae_nonzero'].squeeze()
    dae_mean = results['mean_dae_nonzero'].squeeze()

    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    # Labels for each neuron group
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean and std for each group
    sae_group_means = []
    dae_group_means = []
    sae_group_stds = []
    dae_group_stds = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = sae_mean[start:end]
        dae_group = dae_mean[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        dae_group_stds.append(np.std(dae_group))
    
    # plt.rc('font', size=16)
    
    plt.figure(figsize=(6.288*0.49, 2.5))
    
    x_indices = np.arange(len(neuron_groups))
    width = 0.35
    
    plt.bar(x_indices - width/2, sae_group_means, width, label='AE', color='#1a7adb',
            yerr=sae_group_stds, capsize=4)
    plt.bar(x_indices + width/2, dae_group_means, width, label='Dev-AE', color='#e82817',
            yerr=dae_group_stds, capsize=4)
    
    plt.xlabel('Neuron Group')
    plt.ylabel('Mean Activation')
    # plt.title(f'Mean Non-Zero Activation per Neuron Group ({dataset.upper()})', pad=20)
    
    plt.xticks(x_indices, x_labels, rotation=90)
    if dataset.lower() == "mnist":
        plt.ylim(0, 10)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(bottom=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(f'Results/figures/png/{dataset}_nonzero_activation_per_neuron.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_nonzero_activation_per_neuron.svg', bbox_inches='tight')
    plt.savefig(f'Results/figures/pdf/{dataset}_nonzero_activation_per_neuron.pdf', bbox_inches='tight')
    plt.close()


def plot_zeros_per_neuron(dataset: str = "mnist"):
    """
    Plot the percentage of zero activations per neuron for the SAE and DAE models,
    grouped by neuron ranges and displayed as a grouped bar chart.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    if dataset.lower() == "mnist":
        neuron_groups = [4, 10, 16, 24, 32]
    elif dataset.lower() == "cifar":
        neuron_groups = [6, 10, 16, 28, 48, 90, 128]
        
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    mean_sae_zeros = results['mean_sae_zeros'].squeeze()
    mean_dae_zeros = results['mean_dae_zeros'].squeeze()
    
    start_indices = [0]
    for i in range(1, len(neuron_groups)):
        start_indices.append(neuron_groups[i-1])
    
    # Labels for each neuron group
    x_labels = []
    for start, end in zip(start_indices, neuron_groups):
        x_labels.append(f"{start+1}-{end}")
    
    # Group the neurons and calculate mean for each group
    sae_group_means = []
    dae_group_means = []
    sae_group_stds = []
    dae_group_stds = []
    
    for i, (start, end) in enumerate(zip(start_indices, neuron_groups)):
        sae_group = mean_sae_zeros[start:end]
        dae_group = mean_dae_zeros[start:end]
        
        sae_group_means.append(np.mean(sae_group))
        dae_group_means.append(np.mean(dae_group))
        sae_group_stds.append(np.std(sae_group))
        dae_group_stds.append(np.std(dae_group))
    
    # plt.rc('font', size=16)
    plt.figure(figsize=(6.288*0.49, 2.7))
    
    # Create a grouped bar plot
    x_indices = np.arange(len(neuron_groups))
    width = 0.35
    
    plt.bar(x_indices - width/2, sae_group_means, width, label='AE', color='#1a7adb',
            yerr=sae_group_stds, capsize=4)
    plt.bar(x_indices + width/2, dae_group_means, width, label='Dev-AE', color='#e82817',
            yerr=dae_group_stds, capsize=4)
    
    plt.xlabel('Neuron Group')
    if dataset.lower() == "cifar":
        plt.ylabel('\% Zero Activations ($\\times10^{2}$)')
    else:
        plt.ylabel('\% Zero Activations')

    if dataset.lower() == "cifar":
        plt.yticks([0, 0.005, 0.01, 0.015, 0.02], 
                   ['0', '0.5', '1.0', '1.5', '2.0'])
    # plt.title(f'Neuron Activation Sparsity by Group ({dataset.upper()})', pad=20)
    
    plt.xticks(x_indices, x_labels, rotation=90)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(bottom=0)
    if dataset.lower() == "mnist":
        plt.ylim(0, 40)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.1))
    # plt.tight_layout()
    
    plt.savefig(f'Results/figures/png/{dataset}_zeros_per_neuron.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_zeros_per_neuron.svg', bbox_inches='tight')
    plt.savefig(f'Results/figures/pdf/{dataset}_zeros_per_neuron.pdf', bbox_inches='tight')
    plt.close()


def plot_per_image_zeros_distribution(dataset: str):
    """
    Plot the distribution of percentage of zeros per image for SAE and DAE models.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    
    sae_per_image_zeros = results['sae_per_image_zeros']
    dae_per_image_zeros = results['dae_per_image_zeros']
    
    # plt.rc('font', size=16)
    plt.figure(figsize=(6.288*0.49, 2.5))
    
    plt.hist(sae_per_image_zeros, alpha=0.6, label='AE', color='#1a7adb')
    plt.hist(dae_per_image_zeros, alpha=0.6, label='Dev-AE', color='#e82817')
    
    plt.xlabel('% Zero Activations per Image')
    plt.ylabel('Number of Images')
    # plt.title(f'Neuron Sparsity per Image ({dataset.upper()})', pad=20)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(f'Results/figures/png/{dataset}_per_image_zeros_dist.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_per_image_zeros_dist.svg', bbox_inches='tight')
    plt.savefig(f'Results/figures/pdf/{dataset}_per_image_zeros_dist.pdf', bbox_inches='tight')
    plt.close()


def plot_per_image_activation_distribution(dataset: str):
    """
    Plot the distribution of mean non-zero activations per image for SAE and DAE models.
    
    Args:
        dataset: Dataset used for training ('mnist' or 'cifar')
    """
    result_file = f"Results/{dataset}_bottleneck_activation.npy"
    results = np.load(result_file, allow_pickle=True).item()
    
    sae_per_image_nonzero_means = results['sae_per_image_nonzero_means']
    dae_per_image_nonzero_means = results['dae_per_image_nonzero_means']
    
    # plt.rc('font', size=16)
    plt.figure(figsize=(6.288*0.49, 2.5))
    
    plt.hist(sae_per_image_nonzero_means, alpha=0.6, label='AE', color='#1a7adb')
    plt.hist(dae_per_image_nonzero_means, alpha=0.6, label='Dev-AE', color='#e82817')
    
    plt.xlabel('Mean Activation per Image')
    plt.ylabel('Number of Images')
    # plt.title(f'Non-Zero Neuron Activation per Image ({dataset.upper()})', pad=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(f'Results/figures/png/{dataset}_per_image_nonzero_activation_dist.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_per_image_nonzero_activation_dist.svg', bbox_inches='tight')
    plt.savefig(f'Results/figures/pdf/{dataset}_per_image_nonzero_activation_dist.pdf', bbox_inches='tight')
    plt.close()


def run_bottleneck_activation_analysis(num_models: int, dataset: str, base_path: str):
    """
    Run the complete bottleneck activation analysis.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset used for training ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    compute_bottleneck_activation(num_models, dataset, base_path)
    
    plot_activation_per_neuron(dataset)
    plot_zeros_per_neuron(dataset)
    
    plot_per_image_activation_distribution(dataset)
    plot_per_image_zeros_distribution(dataset)


if __name__ == "__main__":
    num_models = 10
    dataset = "cifar"
    base_path = "/home/david/"
    
    run_bottleneck_activation_analysis(num_models, dataset, base_path)
    
    num_models = 40
    dataset = "mnist"
    run_bottleneck_activation_analysis(num_models, dataset, base_path)
