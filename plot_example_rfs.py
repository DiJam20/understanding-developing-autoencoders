import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import ListedColormap


mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11
})

MODEL_IDX_FOR_EXAMPLE_RF = 1

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

cifar_size_ls = [6, 6, 6, 6, 6, 6, 
                10, 10, 10, 10, 10, 10,
                16, 16, 16, 16, 16, 16,
                28, 28, 28, 28, 28, 28,
                48, 48, 48, 48, 48, 48, 48, 48, 48,
                90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                128, 128, 128, 128, 128, 128, 128, 128, 
                128, 128, 128, 128, 128, 128, 128, 128, 128
                ]

# Neuron group endpoints
mnist_neuron_groups = [4, 10, 16, 24, 32]
cifar_neuron_groups = [6, 10, 16, 28, 48, 90, 128]

def plot_example_rfs(ax, dataset='cifar', epoch_idx=59):
    """
    Plot example receptive fields for the first subplot.
    Show four RFs per group for both AE and Dev-AE in a 2x2 grid.
    Uses original 0-1 clipping and improved spacing with centered titles.
    
    Args:
        ax: Matplotlib axis to plot on
        dataset: Dataset name ('cifar')
        epoch_idx: Epoch index to use
    """
    # Load RFs
    sae_rfs = np.load(f'Results/{dataset}_sae_rfs.npy')
    dae_rfs = np.load(f'Results/{dataset}_dae_rfs.npy')
    
    # Get the neuron indices for each group
    if dataset == 'mnist':
        neuron_groups = mnist_neuron_groups
    elif dataset == 'cifar':
        neuron_groups = cifar_neuron_groups
    
    group_boundaries = [0] + neuron_groups
    
    ax.axis('off')
    
    num_groups = len(group_boundaries) - 1
    
    # Grid with 2 rows: AE, Dev-AE
    outer_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(),
                                                 height_ratios=[1, 1], 
                                                 hspace=0.1)
    
    grid_wspace = 0.15
    rf_wspace = 0.1
    rf_hspace = 0
    
    # Grids for AE and Dev-AE different neuron groups
    ae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                     subplot_spec=outer_grid[0],
                                                     wspace=grid_wspace)
    devae_groups_grid = gridspec.GridSpecFromSubplotSpec(1, num_groups, 
                                                        subplot_spec=outer_grid[1],
                                                        wspace=grid_wspace)
    
    group_labels = []
    for i in range(len(group_boundaries) - 1):
        start_idx = group_boundaries[i]
        end_idx = group_boundaries[i+1]
        group_label = f"{start_idx+1}-{end_idx}"
        group_labels.append(group_label)
    
    row_labels = ["AE", "Dev-AE"]
    
    # Separate axis for the AE and Dev-AE labels
    ae_label_ax = plt.subplot(outer_grid[0])
    ae_label_ax.axis('off')
    ae_label_ax.text(-0.05, 0.5, row_labels[0], rotation=90, 
                     transform=ae_label_ax.transAxes, 
                     ha='center', va='center')
    devae_label_ax = plt.subplot(outer_grid[1])
    devae_label_ax.axis('off')
    devae_label_ax.text(-0.05, 0.5, row_labels[1], rotation=90, 
                        transform=devae_label_ax.transAxes, 
                        ha='center', va='center')
    
    # AE RFs
    for col_group in range(num_groups):
        # Grid with 3 rows: title, 2x2 RFs (2 columns)
        ae_rf_grid = gridspec.GridSpecFromSubplotSpec(3, 2, 
                                                    subplot_spec=ae_groups_grid[col_group],
                                                    height_ratios=[0.2, 1, 1],
                                                    wspace=rf_wspace, hspace=rf_hspace)
        
        start_idx = group_boundaries[col_group]
        
        # Neuron group title
        group_title_ax = plt.subplot(ae_rf_grid[0, :])
        group_title_ax.set_title(group_labels[col_group], fontsize=11, pad=2)
        
        group_title_ax.axis('off')
        
        rf_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for rf_idx, (row, col) in enumerate(rf_positions):
            neuron_idx = start_idx + rf_idx
            
            curr_ax = plt.subplot(ae_rf_grid[row, col])
            
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            
            rf_data = sae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, epoch_idx, neuron_idx]

            if dataset.lower() == 'cifar':
                rf_data = np.transpose(rf_data, (1, 2, 0))
                rf_data = np.clip(rf_data, 0, 1)
            
                curr_ax.imshow(rf_data)

            if dataset == 'mnist':
                rf_data = rf_data.reshape(28, 28)
                curr_ax.imshow(rf_data, cmap='gray')
                
    
    # Dev-AE RFs
    for col_group in range(num_groups):
        # Grid with 3 rows: title, 2x2 RFs (2 columns)
        devae_rf_grid = gridspec.GridSpecFromSubplotSpec(3, 2,
                                                       subplot_spec=devae_groups_grid[col_group],
                                                       height_ratios=[0.2, 1, 1],
                                                       wspace=rf_wspace, hspace=rf_hspace)
        
        start_idx = group_boundaries[col_group]
        
        rf_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        
        for rf_idx, (row, col) in enumerate(rf_positions):
            neuron_idx = start_idx + rf_idx
            
            curr_ax = plt.subplot(devae_rf_grid[row, col])
            
            curr_ax.set_xticks([])
            curr_ax.set_yticks([])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            
            rf_data = dae_rfs[MODEL_IDX_FOR_EXAMPLE_RF, epoch_idx, neuron_idx]

            if dataset.lower() == 'cifar':
                rf_data = np.transpose(rf_data, (1, 2, 0))
                rf_data = np.clip(rf_data, 0, 1)
            
                curr_ax.imshow(rf_data)

            if dataset == 'mnist':
                rf_data = rf_data.reshape(28, 28)
                curr_ax.imshow(rf_data, cmap='gray')
        
    fig.savefig(f"Results/figures/png/{dataset}_example_rfs.png", dpi=300)
    fig.savefig(f"Results/figures/svg/{dataset}_example_rfs.svg")
    fig.savefig(f"Results/figures/pdf/{dataset}_example_rfs.pdf")
    fig.savefig(f"Results/figures/eps/{dataset}_example_rfs.eps")
    

fig = plt.figure(figsize=(6.26*0.72, 1.9))
ax = fig.add_subplot(111)
plot_example_rfs(ax, dataset='mnist')
plt.close(fig)

fig = plt.figure(figsize=(6.26, 1.9))
ax = fig.add_subplot(111)
plot_example_rfs(ax, dataset='cifar')
plt.close(fig)