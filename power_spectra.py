import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Author: Deyue Kong
def z_score(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image using the z-score normalization.
    Normalized image: (pixel - mean) / std

    Args:
        image: 2D numpy array representing the image
    
    Returns:
        normalized_image: 2D numpy array representing the normalized
        image
    """
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image


# Author: Deyue Kong
def radial_profile(data: np.ndarray, center: np.ndarray = None) -> np.ndarray:
    """
    Compute the radial profile of a 2D array data.

    Args:
        data: 2D numpy array representing the image
        center: 1D numpy array representing the center of the image
    
    Returns:
        radialprofile: 1D numpy array representing the radial profile
        of the image
    """
    y, x = np.indices((data.shape))
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())

    radialprofile = tbin / nr

    return radialprofile


# Author: Deyue Kong
def power_spectrum_radial_average(image: np.ndarray) -> np.ndarray:
    """
    Calculate the radial average of the power spectrum for a 2D grey-scale image.
    
    Args:
        image: 2D numpy array representing the image
    Returns:
        radial_avg: radial average of the power spectrum
    """
    # Take the 2D Fourier transform of the image and shift the zero frequency component to the center
    f_transform = np.fft.fftshift(np.fft.fft2(image))

    # Compute the power spectrum (magnitude squared of the Fourier coefficients)
    power_spectrum = np.abs(f_transform) ** 2

    # Compute the radial profile of the power spectrum
    radial_avg = radial_profile(power_spectrum)

    return radial_avg


def rgb_to_grayscale(images):
    """
    Convert RGB images to grayscale.
    
    Parameters:
    images : numpy array of shape (batch_size, 3, height, width)
        Batch of RGB images.
    
    Returns:
    numpy array of shape (batch_size, 1, height, width)
        Batch of grayscale images.
    """
    # Standard luminance formula: 0.299 * R + 0.587 * G + 0.114 * B
    # Source: https://www.w3.org/TR/AERT/#color-contrast
    grayscale = 0.299 * images[:, :, :, 0:1, :] + \
                0.587 * images[:, :, :, 1:2, :] + \
                0.114 * images[:, :, :, 2:3, :]
    
    return np.squeeze(grayscale, axis=3)


def compute_power_spectra(save_path_sae: str, save_path_dae:str, num_models: int, epoch: int) -> tuple:
    """
    Load the receptive fields of the models and compute the power spectrum of each RF.
    
    Args:
        save_path_sae: Path to SAE RF data
        save_path_dae: Path to DAE RF data
        num_models: number of models
        epoch: epoch number
        
    Returns:
        sae_power_spectra: list of power spectra of SAE RFs
        dae_power_spectra: list of power spectra of DAE RFs
    """
    sae_rfs = np.load(save_path_sae)
    dae_rfs = np.load(save_path_dae)

    sae_power_spectra = []
    dae_power_spectra = []

    MIN_WIDTH = 28
    MIN_HEIGHT = 28

    if len(sae_rfs.shape) == 4:
        sae_rfs = sae_rfs.reshape(sae_rfs.shape[:-1] + (28, 28))
        dae_rfs = dae_rfs.reshape(dae_rfs.shape[:-1] + (28, 28))

    if len(sae_rfs.shape) == 6:
        sae_rfs = rgb_to_grayscale(sae_rfs)
        dae_rfs = rgb_to_grayscale(dae_rfs)
        MIN_WIDTH = 32
        MIN_HEIGHT = 32

    for i in tqdm(range(num_models)):
        sae_power_spectrum = []
        for rf in sae_rfs[i, epoch, :]:
            radial_avg = power_spectrum_radial_average(z_score(rf.reshape(MIN_WIDTH, MIN_HEIGHT)))
            sae_power_spectrum.append(radial_avg)
        sae_power_spectra.append(sae_power_spectrum)

        dae_power_spectrum = []
        for rf in dae_rfs[i, epoch, :]:
            radial_avg = power_spectrum_radial_average(z_score(rf.reshape(MIN_WIDTH, MIN_HEIGHT)))
            dae_power_spectrum.append(radial_avg)
        dae_power_spectra.append(dae_power_spectrum)

    return sae_power_spectra, dae_power_spectra


def group_power_spectra(power_spectra, neuron_groups):
    """
    Group power spectra by neuron groups and calculate the average for each group.
    
    Args:
        power_spectra: Power spectra for each neuron, shape (n_neurons, n_frequencies)
        neuron_groups: List of integers representing the end index of each group
                      e.g., [6, 10, 16, 28, 90, 128] means groups are 1-6, 7-10, etc.
    
    Returns:
        grouped_spectra: List of averaged power spectra for each group
        group_labels: Labels for each group (e.g., "1-6", "7-10", etc.)
    """
    grouped_spectra = []
    group_labels = []
    
    start_idx = 0
    for end_idx in neuron_groups:
        end_idx = min(end_idx, power_spectra.shape[0])
        
        # Calculate average power spectrum for one group
        group_avg = np.mean(power_spectra[start_idx:end_idx], axis=0)
        grouped_spectra.append(group_avg)
        
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
        
        # Update the start index for the next group
        start_idx = end_idx
    
    return np.array(grouped_spectra), group_labels


def plot_power_spectra_subplot(ax, frequency_data, title, group_labels=None, ylim_top=30000, show_y_label=True):
    """
    Plot power spectra for groups of neurons.
    
    Args:
        ax: Matplotlib axis
        frequency_data: Power spectra data, shape (n_groups, n_frequencies)
        title: Plot title
        group_labels: Labels for each group
        ylim_top: Upper limit for y-axis
        show_y_label: Whether to show y label and ticks
    """
    colors = plt.cm.cool(np.linspace(0, 1, frequency_data.shape[0]))
    
    if group_labels is not None:
        for idx, freq in enumerate(frequency_data):
            label = f'Neurons {group_labels[idx]}' if group_labels else f'RF{idx + 1}'
            ax.plot(freq, color=colors[idx], label=label, linewidth=2)

    else:
        for idx, freq in enumerate(frequency_data):
            ax.plot(freq, color=colors[idx], label=f'RF{idx + 1}')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ylim_top)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '5', '10'])
    ax.set_xlabel('Frequency')
    
    if show_y_label:
        ax.set_ylabel('Power')
    else:
        ax.set_yticks([])
        ax.set_ylabel('')
    
    ax.tick_params(axis='both', which='major')
    ax.set_title(title, fontsize=11)


def plot_power_spectra(model_type, sae_power_spectra, dae_power_spectra, neuron_groups=None, save_path="Results/figures"):
    """
    Plot the power spectra of the receptive fields.
    
    If neuron_groups is provided, plots grouped and averaged power spectra with error bands.
    If neuron_groups is None, plots individual power spectra for each neuron.

    Args:
        model_type: Model type ('mnist' or 'cifar')
        sae_power_spectra: Power spectra for SAE neurons
        dae_power_spectra: Power spectra for DAE neurons
        neuron_groups: List of integers representing the end index of each group
                      If None, each neuron's spectrum is plotted individually.
        save_path: Directory to save the plots
    """
    ylim_top = 15000 if model_type == 'mnist' else 30000
    
    # Average across models
    sae_mean = np.mean(sae_power_spectra, axis=0)
    dae_mean = np.mean(dae_power_spectra, axis=0)
    
    # Individual neuron plotting
    if neuron_groups is None:
        ylim_top = 16000 if model_type == 'mnist' else 30000

        fig = plt.figure(figsize=(6.266*0.8, 1.5), dpi=300)
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
        
        ax_sae = plt.subplot(gs[0])
        ax_dae = plt.subplot(gs[1])
        ax_cbar = plt.subplot(gs[2])
        
        num_neurons = sae_mean.shape[0]
        cmap = plt.cm.cool
        colors = cmap(np.linspace(0, 0.8, num_neurons))
        
        for i in range(num_neurons):
            freq_data = sae_mean[i][:15]
            ax_sae.plot(freq_data, color=colors[i], linewidth=0.5, alpha=0.7)
            
            freq_data = dae_mean[i][:15]
            ax_dae.plot(freq_data, color=colors[i], linewidth=0.5, alpha=0.7)
            
        # Configure axes
        for ax, title in [(ax_sae, 'AE'), (ax_dae, 'Dev-AE')]:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, ylim_top)
            ax.set_xticks([1, 5, 9])
            ax.set_xticklabels(['1\nLow', '5\nMedium', '9\nHigh'])
            ax.set_xlabel('Frequency')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(title, fontsize=11)
        
        # Set y-ticks for each axis
        if model_type == 'mnist':
            ax_sae.set_yticks([5000, 10000, 15000])
            ax_sae.set_yticklabels([0.5, 1.0, 1.5])
            ax_dae.set_yticks([5000, 10000, 15000])
            ax_dae.set_yticklabels(['', '', ''])
        else:
            ax_sae.set_yticks([10000, 20000, 30000])
            ax_sae.set_yticklabels([1, 2, 3])
            ax_dae.set_yticks([10000, 20000, 30000])
            ax_dae.set_yticklabels(['', '', ''])
        
        ax_sae.set_ylabel('Power ($\\times10^4$)')
        
        # Create colorbar for individual neurons
        norm = plt.Normalize(0, num_neurons-1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=ax_cbar)
        cbar.set_label("Neuron index")
        
        if model_type == 'mnist':
            cbar.set_ticks([0.5, 4.5, 10.5, 16.5, 24.5])
            cbar.set_ticklabels([1, 5, 11, 17, 25])
        elif model_type == 'cifar':
            cbar.set_ticks([0.5, 16.5, 28.5, 48.5, 90.5])
            cbar.set_ticklabels([1, 17, 29, 49, 91])
        
        # Save with individual suffix
        plt.savefig(f"{save_path}/png/{model_type}_power_spectrum_individual.png", dpi=300)
        plt.savefig(f"{save_path}/svg/{model_type}_power_spectrum_individual.svg")
        plt.savefig(f"{save_path}/pdf/{model_type}_power_spectrum_individual.pdf", bbox_inches='tight')
        plt.savefig(f"{save_path}/eps/{model_type}_power_spectrum_individual.eps")
        plt.close()
        return
        
    # Grouped plotting
    fig = plt.figure(figsize=(6.266*0.8, 1.5), dpi=300)
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    
    # Create axes
    ax_sae = plt.subplot(gs[0])
    ax_dae = plt.subplot(gs[1])
    ax_cbar = plt.subplot(gs[2])
    
    # Process neuron groups
    group_boundaries = [0] + neuron_groups
    group_labels = []
    sae_grouped_means = []
    sae_grouped_stds = []
    dae_grouped_means = []
    dae_grouped_stds = []
    
    # Calculate group statistics
    for i in range(len(group_boundaries) - 1):
        start_idx = group_boundaries[i]
        end_idx = group_boundaries[i+1]
        
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
        
        # SAE
        group_data = sae_mean[start_idx:end_idx]
        group_mean = np.mean(group_data, axis=0)
        group_std = np.std(group_data, axis=0)
        sae_grouped_means.append(group_mean)
        sae_grouped_stds.append(group_std)
        
        # DAE
        group_data = dae_mean[start_idx:end_idx]
        group_mean = np.mean(group_data, axis=0)
        group_std = np.std(group_data, axis=0)
        dae_grouped_means.append(group_mean)
        dae_grouped_stds.append(group_std)
    
    n_groups = len(group_labels)
    colors = plt.cm.cool(np.linspace(0, 1, n_groups))
    
    for idx in range(n_groups):
        freq_data = sae_grouped_means[idx][:15]
        std_data = sae_grouped_stds[idx][:15]
        ax_sae.plot(freq_data, color=colors[idx], linewidth=2)
        ax_sae.fill_between(
            range(len(freq_data)),
            freq_data - std_data,
            freq_data + std_data,
            color=colors[idx],
            alpha=0.2
        )
    
    for idx in range(n_groups):
        freq_data = dae_grouped_means[idx][:15]
        std_data = dae_grouped_stds[idx][:15]
        ax_dae.plot(freq_data, color=colors[idx], linewidth=2)
        ax_dae.fill_between(
            range(len(freq_data)),
            freq_data - std_data,
            freq_data + std_data,
            color=colors[idx],
            alpha=0.2
        )
    
    # Configure axes
    for ax, title in [(ax_sae, 'AE'), (ax_dae, 'Dev-AE')]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, ylim_top)
        ax.set_xticks([1, 5, 9])
        ax.set_xticklabels(['1\nLow', '5\nMedium', '9\nHigh'])
        ax.set_xlabel('Frequency')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=11)
    
    # Set y-ticks for each axis
    if model_type == 'mnist':
        ax_sae.set_yticks([5000, 10000, 15000])
        ax_sae.set_yticklabels([0.5, 1.0, 1.5])
        ax_dae.set_yticks([5000, 10000, 15000])
        ax_dae.set_yticklabels(['', '', ''])
    else:
        ax_sae.set_yticks([10000, 20000, 30000])
        ax_sae.set_yticklabels([1, 2, 3])
        ax_dae.set_yticks([10000, 20000, 30000])
        ax_dae.set_yticklabels(['', '', ''])
    
    ax_sae.set_ylabel('Power ($\\times10^4$)')
    
    # Create colorbar
    colors_reversed = plt.cm.cool(np.linspace(0, 1, n_groups))[::-1]
    group_labels_reversed = group_labels[::-1]
    discrete_cmap = mcolors.ListedColormap(colors_reversed)
    bounds = np.arange(-0.5, n_groups + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    tick_positions = np.arange(n_groups)
    
    cbar = plt.colorbar(sm, cax=ax_cbar, ticks=tick_positions)
    cbar.set_label("Neuron Group")
    cbar.set_ticklabels(group_labels_reversed)
    cbar.minorticks_off()
    
    # Save with grouped suffix
    plt.savefig(f"{save_path}/png/{model_type}_power_spectrum_grouped.png", dpi=300)
    plt.savefig(f"{save_path}/svg/{model_type}_power_spectrum_grouped.svg")
    plt.savefig(f"{save_path}/pdf/{model_type}_power_spectrum_grouped.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}/eps/{model_type}_power_spectrum_grouped.eps")
    plt.close()


def analyze_power_spectra(model_type, save_path_sae, save_path_dae, num_models, epoch, neuron_groups=None):
    """
    Load RFs, compute power spectra, and plot them with optional neuron grouping.
    
    Args:
        save_path_sae: Path to SAE RF data
        save_path_dae: Path to DAE RF data
        num_models: Number of models
        epoch: Epoch number
        neuron_groups: List of integers representing the end index of each group
                      e.g., [6, 10, 16, 28, 90, 128] means groups are 1-6, 7-10, etc.
    """
    sae_power_spectra, dae_power_spectra = compute_power_spectra(save_path_sae, save_path_dae, num_models, epoch)
    plot_power_spectra(model_type, sae_power_spectra, dae_power_spectra, neuron_groups)


if __name__ == "__main__":
    analyze_power_spectra(
        model_type='mnist',
        save_path_sae='Results/mnist_sae_rfs.npy',
        save_path_dae='Results/mnist_dae_rfs.npy',
        num_models=40,
        epoch=59,
        neuron_groups=[4, 10, 16, 24, 32]
    )

    analyze_power_spectra(
        model_type='cifar',
        save_path_sae='Results/cifar_sae_rfs.npy',
        save_path_dae='Results/cifar_dae_rfs.npy',
        num_models=10,
        epoch=59,
        neuron_groups=[6, 10, 16, 28, 48, 90, 128]
    )