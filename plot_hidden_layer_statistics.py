import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autoencoder import *
from model_utils import *
from solver import *


def load_hidden_layer_statistics(model_type: str, dataset: str) -> tuple:
    hidden_layer_results_path = f"Results/hidden_layer_act_{model_type}_{dataset}.npy"
    neuron_activations = np.load(hidden_layer_results_path)

    # Get the mean activations (index 0) for the specified epoch
    # neuron_activations shape: (num_models, num_epochs, num_layers, 4 statistics values, neurons_per_layer)
    mean_activations = neuron_activations[:, :, 0, :]
    model_averaged = np.nanmean(mean_activations, axis=0)
    layer_averages = np.nanmean(model_averaged, axis=1)

    # Get the standard deviation of activations (index 1) for the specified epoch
    mean_activations_std = neuron_activations[:, :, 1, :]
    model_averaged_std = np.nanmean(mean_activations_std, axis=0)
    layer_std = np.nanmean(model_averaged_std, axis=1)

    return layer_averages, layer_std


def load_zero_activation_statistics(model_type: str, dataset: str) -> tuple:
    """
    Load zero activation statistics for the specified model type and architecture.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset type ('mnist' or 'cifar')
                
    Returns:
        tuple: Statistics about zero activations categories
    """
    hidden_layer_results_path = f"Results/hidden_layer_act_{model_type}_{dataset}.npy"
    neuron_activations = np.load(hidden_layer_results_path)
    
    always_zero_activations = neuron_activations[:, :, 4, :]
    model_averaged_always = np.nanmean(always_zero_activations, axis=0)
    always_zero_averages = np.nanmean(model_averaged_always, axis=1)
    
    never_zero_activations = neuron_activations[:, :, 5, :]
    model_averaged_never = np.nanmean(never_zero_activations, axis=0)
    never_zero_averages = np.nanmean(model_averaged_never, axis=1)
    
    sometimes_zero_activations = neuron_activations[:, :, 6, :]
    model_averaged_sometimes = np.nanmean(sometimes_zero_activations, axis=0)
    sometimes_zero_averages = np.nanmean(model_averaged_sometimes, axis=1)
    
    return always_zero_averages, never_zero_averages, sometimes_zero_averages


def plot_neuron_activations(dataset: str) -> None:
    """
    Plot activations given a specific model architecture with two subplots:
    left for SAE statistics and right for DAE statistics.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset type ('mnist' or 'cifar')
        
    Returns:
        None: Saves plot to file
    """    
    sae_layer_averages, sae_layer_std = load_hidden_layer_statistics('sae', dataset)
    dae_layer_averages, dae_layer_std = load_hidden_layer_statistics('dae', dataset)

    if dataset == 'cifar':
        sae_layer_averages = sae_layer_averages[:-1]
        sae_layer_std = sae_layer_std[:-1]
        dae_layer_averages = dae_layer_averages[:-1]
        dae_layer_std = dae_layer_std[:-1]

        sae_layer_averages = np.concatenate([sae_layer_averages[:5], sae_layer_averages[6:]])
        sae_layer_std = np.concatenate([sae_layer_std[:5], sae_layer_std[6:]])
        dae_layer_averages = np.concatenate([dae_layer_averages[:5], dae_layer_averages[6:]])
        dae_layer_std = np.concatenate([dae_layer_std[:5], dae_layer_std[6:]])
    
    # plt.rc('font', size=20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    if dataset == 'mnist':
        x_labels = ['Enc1', 'Enc2', 'Bottleneck', 'Dec1', 'Dec2']
    elif dataset == 'cifar':
        x_labels = [
            'Enc 1', 'Enc 2', 'Enc 3', 'Enc 4', 'Enc 5',
            'Bottleneck', 'Dec 1', 'Dec 2',
            'Dec 3', 'Dec 4'
        ]

    # Plot SAE
    x_indices = np.arange(len(sae_layer_averages))
    sae_bars = ax1.bar(x_indices, sae_layer_averages, color='#1a7adb', yerr=sae_layer_std, capsize=5)
    ax1.set_xticks(x_indices)
    if dataset == 'cifar':
        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Activation')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot DAE
    x_indices = np.arange(len(dae_layer_averages))
    dae_bars = ax2.bar(x_indices, dae_layer_averages, color='#e82817', yerr=dae_layer_std, capsize=5)
    ax2.set_xticks(x_indices)
    if dataset == 'cifar':
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax2.set_xticklabels(x_labels)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')


    ax2.legend([sae_bars, dae_bars], ['AE', 'Dev-AE'], loc='upper right')

    max_val = max(
        max(sae_layer_averages) + max(sae_layer_std),
        max(dae_layer_averages) + max(dae_layer_std)
    )
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    # fig.suptitle(f'Hidden Layer Activations ({dataset.upper()})', fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    

    plt.savefig(f"Results/figures/png/hidden_layer_activations_{dataset}.png")
    plt.savefig(f"Results/figures/svg/hidden_layer_activations_{dataset}.svg")
    plt.savefig(f"Results/figures/pdf/hidden_layer_activations_{dataset}.pdf")
    plt.close()
    
    return None


def plot_zero_activations(dataset: str) -> None:
    """
    Plot zero activations as a stacked bar chart showing always zero, never zero and sometimes zero percentages.
    
    Args:
        dataset: Dataset type ('mnist' or 'cifar')
        
    Returns:
        None: Saves plot to file
    """    
    sae_always_zero, sae_never_zero, sae_sometimes_zero = load_zero_activation_statistics('sae', dataset)
    dae_always_zero, dae_never_zero, dae_sometimes_zero = load_zero_activation_statistics('dae', dataset)
    
    if dataset == 'cifar':
        # Remove the last element (output layer)
        sae_always_zero = sae_always_zero[:-1]
        sae_never_zero = sae_never_zero[:-1]
        sae_sometimes_zero = sae_sometimes_zero[:-1]
        
        dae_always_zero = dae_always_zero[:-1]
        dae_never_zero = dae_never_zero[:-1]
        dae_sometimes_zero = dae_sometimes_zero[:-1]

        sae_always_zero = np.concatenate([sae_always_zero[:5], sae_always_zero[6:]])
        sae_never_zero = np.concatenate([sae_never_zero[:5], sae_never_zero[6:]])
        sae_sometimes_zero = np.concatenate([sae_sometimes_zero[:5], sae_sometimes_zero[6:]])
        
        dae_always_zero = np.concatenate([dae_always_zero[:5], dae_always_zero[6:]])
        dae_never_zero = np.concatenate([dae_never_zero[:5], dae_never_zero[6:]])
        dae_sometimes_zero = np.concatenate([dae_sometimes_zero[:5], dae_sometimes_zero[6:]])
    
    # plt.rc('font', size=20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.268, 2.5), dpi=300)
    
    if dataset == 'mnist':
        x_labels = ['Encoder (1)', 'Encoder (2)', 'Bottleneck', 'Decoder (1)', 'Decoder (2)']
    elif dataset == 'cifar':
        x_labels = [
            'Encoder (1)', 'Encoder (2)', 'Encoder (3)', 'Encoder (4)', 'Encoder (5)',
            'Bottleneck', 'Decoder (1)', 'Decoder (2)',
            'Decoder (3)', 'Decoder (4)'
        ]
    
    # Plot SAE
    x_indices = np.arange(len(sae_always_zero))
    
    sae_bottom_sometimes = sae_always_zero
    sae_bottom_never = sae_always_zero + sae_sometimes_zero
    
    bar1 = ax1.bar(x_indices, sae_always_zero, color='#D72638')
    bar2 = ax1.bar(x_indices, sae_sometimes_zero, bottom=sae_bottom_sometimes, color='#e8b81c')
    bar3 = ax1.bar(x_indices, sae_never_zero, bottom=sae_bottom_never, color='#1B998B')
    
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=90)
    ax1.set_ylabel('Activity Distribution (\%)')
    ax1.set_ylim(0, 100)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('AE', fontsize=11)
    
    # Plot DAE
    x_indices = np.arange(len(dae_always_zero))
    
    dae_bottom_sometimes = dae_always_zero
    dae_bottom_never = dae_always_zero + dae_sometimes_zero
    
    ax2.bar(x_indices, dae_always_zero, color='#D72638')
    ax2.bar(x_indices, dae_sometimes_zero, bottom=dae_bottom_sometimes, color='#e8b81c')
    ax2.bar(x_indices, dae_never_zero, bottom=dae_bottom_never, color='#1B998B')
    
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, rotation=90)
    ax2.set_ylim(0, 100)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')
    ax2.set_title('Dev-AE', fontsize=11)

    fig.legend([bar1, bar2, bar3], 
              ['Inactive\nNeurons', 'Conditionally\nActive\nNeurons', 'Universally\nActive\nNeurons'], 
              loc='center', 
              bbox_to_anchor=(0.543, 0.6),
              frameon=True,
              fontsize=11,
              labelspacing=0.5)
    
    # fig.suptitle(f'Neuron Activation Sparsity Across Network Layers ({dataset.upper()})', fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.8)
    
    plt.savefig(f"Results/figures/png/neuron_activation_patterns_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/neuron_activation_patterns_{dataset}.svg", bbox_inches='tight')
    plt.savefig(f"Results/figures/pdf/neuron_activation_patterns_{dataset}.pdf", bbox_inches='tight')
    plt.close()
    
    return None


def activations_heatmap(model_type: str, layer_idx: int = 2) -> None:
    """
    DEPRECATED (only works for MNIST)
    Generate a heatmap of neuron activations over epochs.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        layer_idx: Index of the layer to visualize (default: 2, which is the bottleneck layer)

    Returns:
        None: Saves plot to file
    """
    neurons_per_layer = [512, 128, 32, 128, 512]

    neuron_activations = np.load(f"Results/{model_type}_hidden_layer_neuron_activations.npy")
    epoch_activations_mean = np.nanmean(neuron_activations[:, :, layer_idx, 0, :], axis=0)

    num_neurons = neurons_per_layer[layer_idx]
    epoch_activations_mean = epoch_activations_mean[:, :num_neurons]

    epoch_activations_mean = epoch_activations_mean.T
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    num_neurons = epoch_activations_mean.shape[0]
    
    heatmap = sns.heatmap(
        epoch_activations_mean,
        cmap="inferno",
        vmin=0,
        vmax=10,
        cbar_kws={"label": "Angle between PCs"},
        linewidths=0.5,
        square=True
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Strength of Activation", fontsize=24)
    cbar.minorticks_off()

    # Set x-axis ticks (epochs)
    num_epochs = max(epoch_activations_mean.shape[1], 30)
    ax.set_xticks([0.5, num_epochs//2 - 0.5, num_epochs - 0.5])
    ax.set_xticklabels(["1", str(num_epochs//2), str(num_epochs)], fontsize=11, rotation=0)
    
    # Set y-axis ticks (neurons)
    y_ticks = [0.5, num_neurons//2 - 0.5, num_neurons - 0.5]
    y_labels = ["1", str(num_neurons//2), str(num_neurons)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, rotation=90)

    # ax.set_title(f"Neuron Activation over Epochs ({model_type})", fontsize=28, pad=25)
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Neuron Index", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"Results/{model_type}_neuron_activations_over_time.png")
    plt.close()
    
    return None


def plot_hidden_layer_activation(dataset: str) -> None:
    """
    Plot hidden layer activations for the specified model architecture.

    Args:
        dataset: Dataset type ('mnist' or 'cifar')
    
    Returns:
        None: Saves plot to file
    """
    plot_neuron_activations(dataset=dataset)
    plot_zero_activations(dataset=dataset)
    return None