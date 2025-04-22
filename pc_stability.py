import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns
import torch
from sklearn.decomposition import PCA
import concurrent.futures
import multiprocessing
from tqdm import tqdm

from model_utils import *

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11,
    'axes.titlesize': 11
})


def calculate_angles_for_single_model(model_idx, model_type, test_images, dataset='mnist', 
                                      compare_final_epoch=False, size_ls=None, 
                                      num_epochs=60, base_path='/home/david/'):
    """
    Calculate PCA angles for a single model across epochs.
    
    Args:
        model_idx: Index of the model to process
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset to use ('mnist' or 'cifar')
        compare_final_epoch: Compare final epoch or between epochs
        size_ls: List of component sizes for DAE
        num_epochs: Number of epochs to process
        
    Returns:
        list: Angles between PCs for this model
    """
    if dataset.lower() == 'mnist':
        model_path = f"{base_path}mnist_models/{model_type}/{model_idx}"
        num_components = 32
        loader_function = load_model
    elif dataset.lower() == 'cifar':
        model_path = f"{base_path}cifar_models/{model_type}/{model_idx}"
        num_components = 128
        loader_function = load_conv_model
    
    latent_matrices = []
    
    # Calculate latent matrices for each epoch
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        latent_matrix = []
        ae = loader_function(model_path, model_type=model_type, epoch=epoch, size_ls=size_ls)
        
        for image in test_images:
            if dataset.lower() == 'mnist':
                # Process MNIST images
                image = torch.tensor(image, dtype=torch.float32).reshape(-1)
                with torch.no_grad():
                    encoded, _ = ae(image)
                    latent_matrix.append(encoded.detach().numpy())
            else:
                # Process CIFAR images
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    encoded, _ = ae(image)
                    encoded = torch.squeeze(encoded)
                    latent_matrix.append(encoded.detach().numpy())
        
        latent_matrix = np.stack(latent_matrix)
        
        # Perform PCA on all available bottleneck neurons
        if model_type == "sae":
            pca = PCA(n_components=num_components)
            pca.fit(latent_matrix)
            pca_components = np.array(pca.components_)
        elif model_type == "dae":
            pca = PCA(n_components=size_ls[epoch])
            pca.fit(latent_matrix)
            pca_components = np.pad(pca.components_, (0, num_components - pca.components_.shape[0]), 'constant')
            # pca_components = np.zeros((num_components, pca.components_.shape[1]))
            # components_to_use = min(num_components, pca.components_.shape[0])
            # pca_components[:components_to_use] = pca.components_[:components_to_use]
            
        latent_matrices.append(pca_components)
    
    # Calculate angles based on comparison method
    angles_per_model = []
    for i in range(num_components):
        angles_per_pc = []
        
        if compare_final_epoch:
            # Compare each epoch with the final epoch
            final_epoch_vector = latent_matrices[-1][i]
            for j in range(num_epochs-1):
                cosine_angle = cosine_angle_between_vectors(latent_matrices[j][i], final_epoch_vector)
                angles_per_pc.append(cosine_angle)
        else:
            # Compare consecutive epochs
            for j in range(num_epochs-1):
                cosine_angle = cosine_angle_between_vectors(latent_matrices[j][i], latent_matrices[j+1][i])
                angles_per_pc.append(cosine_angle)
                
        angles_per_model.append(angles_per_pc)
    
    return angles_per_model


def compute_angle_matrix(model_type, dataset='mnist', compare_final_epoch=False, 
                         size_ls=None, num_models=10, 
                         num_epochs=60):
    """
    Calculate principal component angles for multiple models using parallel processing.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset to use ('mnist' or 'cifar')
        compare_final_epoch: Compare final epoch or between epochs
        size_ls: List of component sizes for DAE models
        num_models: Number of models to process
        num_epochs: Number of epochs per model
        
    Returns:
        None: Results are saved to a file
    """    
    # Get number of CPU cores (leave one free for system processes)
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    comparison_type = "final_epoch" if compare_final_epoch else "consecutive"

    result_file = f"Results/{dataset}_{model_type}_pc_stability_{comparison_type}_angles.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    if dataset.lower() == 'mnist':
        test_images, _ = load_mnist_list()
    elif dataset.lower() == 'cifar':
        test_images, _ = load_cifar_list()
        test_images = test_images[:1000]
    
    # Create process pool and map the function across all models
    # Source: "Speed Up Your Python Program With Concurrency" by Jim Anderson
    # Date: November 25, 2024
    # URL: https://jimanderson.dev/python-concurrency/
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in range(num_models):
            future = executor.submit(calculate_angles_for_single_model, idx, 
                                     model_type, test_images, dataset, 
                                     compare_final_epoch, size_ls, num_epochs)
            futures.append(future)
            
        all_angles = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_models, desc=f"Processing {model_type} models"):
            result = future.result()
            all_angles.append(result)
    
    np.save(result_file, all_angles)
    print(f"Results saved to {result_file}")


def compute_average_angle_matrix(model_type, dataset='mnist', compare_final_epoch=False):
    comparison_type = "final_epoch" if compare_final_epoch else "consecutive"
    all_angles = np.load(f"Results/{dataset}_{model_type}_pc_stability_{comparison_type}_angles.npy")
    
    if model_type == "sae":
        average_all_angles = np.mean(all_angles, axis=0)
        angle_matrix = np.array(average_all_angles)
        non_computable_cells = np.zeros_like(angle_matrix)
    elif model_type == "dae":
        average_all_angles = np.mean(all_angles, axis=0)
        angle_matrix = np.array(average_all_angles)
        highlighted_non_computable_angles = angle_matrix.copy()

        for i in range(highlighted_non_computable_angles.shape[0]):
            for j in range(highlighted_non_computable_angles.shape[1] - 1):
                if np.isnan(highlighted_non_computable_angles[i, j]) and not np.isnan(highlighted_non_computable_angles[i, j + 1]):
                    highlighted_non_computable_angles[i, j] = 100

        mask = highlighted_non_computable_angles == 100
        non_computable_cells = np.where(mask, 1, np.nan)

    return angle_matrix, non_computable_cells


def create_heatmap(angle_matrix, non_computable_cells, model_type, dataset='mnist', compare_final_epoch=False):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    heatmap = sns.heatmap(
        angle_matrix[:, :],
        cmap="plasma",
        vmin=0,
        vmax=90,
        cbar_kws={"label": "Angle between PCs"},
        square=True,
    )

    # Grey out non-computable cells
    if model_type == "dae":
        cmap_grey = ListedColormap(['grey'])
        sns.heatmap(
            non_computable_cells[:, :],
            cmap=cmap_grey,
            cbar=False,
            alpha=1,
            square=True,
        )

    # Customize colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0°", "45°", "90°"], fontsize=24)
    cbar.set_label("PC Angle Difference", fontsize=24)
    cbar.minorticks_off()

    # Customize axes
    max_epochs = angle_matrix.shape[1]
    mid_epoch = max_epochs // 2
    
    if compare_final_epoch:
        ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
        ax.set_xticklabels(["1-Final", f"{mid_epoch}-Final", f"{max_epochs-1}-Final"], 
                            fontsize=24, rotation=0)
    else:
        ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
        ax.set_xticklabels(["1-2", f"{mid_epoch}-{mid_epoch+1}", f"{max_epochs}-{max_epochs+1}"], 
                            fontsize=24, rotation=0)

    num_pcs = angle_matrix.shape[0]
    mid_pc = num_pcs // 2
    ax.set_yticks([0.5, mid_pc - 0.5, num_pcs - 0.5])
    ax.set_yticklabels(["1", f"{mid_pc}", f"{num_pcs}"], fontsize=24, rotation=90)
    
    if model_type == "sae":
        ax.set_title(f"Stability of Principal Components (AE, {dataset.upper()})", fontsize=28, pad=25)
    elif model_type == "dae":
        ax.set_title(f"Stability of Principal Components (DevAE, {dataset.upper()})", fontsize=28, pad=25)
    
    ax.set_xlabel("Epoch Comparison", fontsize=24)
    ax.set_ylabel("Principal Component Index", fontsize=24)

    comparison_type = "vs_final" if compare_final_epoch else "consecutive"
    plt.savefig(f"Results/figures/png/{dataset}_{model_type}_stability_of_pcs_{comparison_type}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"Results/figures/svg/{dataset}_{model_type}_stability_of_pcs_{comparison_type}.svg", bbox_inches="tight")
    plt.close(fig)


def analyze_pc_stability(model_type, dataset='mnist', compare_final_epoch=False, size_ls=None, num_models=10, num_epochs=60):
    """
    Main function to perform PC stability analysis for a single model type.
    
    Args:
        model_type: Type of model ('sae' or 'dae')
        dataset: Dataset to use ('mnist' or 'cifar')
        compare_final_epoch: Compare final epoch or between epochs
        size_ls: List of component sizes for DAE models (None for SAE)
        num_models: Number of models to process
        num_epochs: Number of epochs per model
    """
    compute_angle_matrix(model_type, dataset, compare_final_epoch, size_ls, num_models, num_epochs)
    angle_matrix, non_computable_cells = compute_average_angle_matrix(model_type, dataset, compare_final_epoch)
    
    # Store results for combined visualization
    comparison_type = "final_epoch" if compare_final_epoch else "consecutive"
    np.savez(f"Results/{dataset}_{model_type}_pc_stability_{comparison_type}_processed.npz", 
             angle_matrix=angle_matrix, 
             non_computable_cells=non_computable_cells)
    
    print(f"PC stability analysis for {model_type} on {dataset} complete.")


def create_combined_stability_heatmaps(dataset='mnist', compare_final_epoch=False):
    """
    Create combined heatmaps for SAE and DAE PC stability with a shared colorbar.
    This function should be called after running analyze_pc_stability for both SAE and DAE.
    
    Args:
        dataset: Dataset ('mnist' or 'cifar')
        compare_final_epoch: Whether angles were computed against final epoch
    """
    
    comparison_type = "final_epoch" if compare_final_epoch else "consecutive"
    
    sae_data = np.load(f"Results/{dataset}_sae_pc_stability_{comparison_type}_processed.npz")
    dae_data = np.load(f"Results/{dataset}_dae_pc_stability_{comparison_type}_processed.npz")
    
    sae_angle_matrix = sae_data['angle_matrix']
    sae_non_computable = sae_data['non_computable_cells']
    dae_angle_matrix = dae_data['angle_matrix']
    dae_non_computable = dae_data['non_computable_cells']
    
    fig = plt.figure(figsize=(6.266, 1.7), dpi=300)
    
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    
    ax_sae = plt.subplot(gs[0])
    ax_dae = plt.subplot(gs[1])
    ax_cbar = plt.subplot(gs[2])
    
    ax_sae.set_aspect('auto')
    ax_dae.set_aspect('auto')
    
    cmap = plt.cm.plasma
    norm = plt.Normalize(0, 90)
    
    # SAE heatmap
    sns.heatmap(
        sae_angle_matrix[:, :],
        cmap=cmap,
        vmin=0,
        vmax=90,
        cbar=False,
        ax=ax_sae
    )
    
    # DAE heatmap
    sns.heatmap(
        dae_angle_matrix[:, :],
        cmap=cmap,
        vmin=0,
        vmax=90,
        cbar=False,
        ax=ax_dae
    )
    
    # Add gray to non-computable areas in DAE heatmap if comparing consecutive
    if not compare_final_epoch:
        if np.any(~np.isnan(dae_non_computable)):
            cmap_grey = ListedColormap(['grey'])
            sns.heatmap(
                dae_non_computable[:, :],
                cmap=cmap_grey,
                cbar=False,
                alpha=1,
                ax=ax_dae
            )
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0", "45", "90"])
    cbar.set_label("Cosine Angle Difference (°)", labelpad=10)
    cbar.outline.set_linewidth(0.815)
    cbar.outline.set_edgecolor('black')
    cbar.minorticks_off()
    
    ax_sae.set_title("AE", pad=10)
    ax_dae.set_title("Dev-AE", pad=10)
    
    max_epochs = sae_angle_matrix.shape[1]
    mid_epoch = max_epochs // 2
    
    for ax in [ax_sae, ax_dae]:
        if compare_final_epoch:
            ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
            ax.set_xticklabels(["1", f"{mid_epoch}", f"{max_epochs-1}"])
        else:
            ax.set_xticks([0.5, mid_epoch - 0.5, max_epochs - 0.5])
            ax.set_xticklabels(["1", f"{mid_epoch}", f"{max_epochs}"])
        ax.set_xlabel("Epoch")
    
    num_pcs = sae_angle_matrix.shape[0]
    mid_pc = num_pcs // 2
    
    ax_sae.set_yticks([0.5, mid_pc - 0.5, num_pcs - 0.5])
    ax_sae.set_yticklabels(["1", f"{mid_pc}", f"{num_pcs}"], rotation=0)
    ax_sae.set_ylabel("Principal Component Index")
    
    ax_dae.set_yticks([])
    ax_dae.set_yticklabels([])
    ax_dae.set_ylabel("")
    
    plt.savefig(f"Results/figures/png/{dataset}_stability_of_pcs_{comparison_type}.png", 
                bbox_inches="tight", dpi=300)
    plt.savefig(f"Results/figures/svg/{dataset}_stability_of_pcs_{comparison_type}.svg", 
                bbox_inches="tight")
    plt.savefig(f"Results/figures/pdf/{dataset}_stability_of_pcs_{comparison_type}.pdf", 
                bbox_inches="tight")
    plt.close(fig)


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

    # analyze_pc_stability(model_type='sae', dataset='mnist', compare_final_epoch=True, size_ls=None, num_models=40, num_epochs=60)
    # analyze_pc_stability(model_type='dae', dataset='mnist', compare_final_epoch=True, size_ls=mnist_size_ls, num_models=40, num_epochs=60)
    # analyze_pc_stability(model_type='sae', dataset='cifar', compare_final_epoch=True, size_ls=None, num_models=10, num_epochs=60)
    # analyze_pc_stability(model_type='dae', dataset='cifar', compare_final_epoch=True, size_ls=cifar_size_ls, num_models=10, num_epochs=60)

    # analyze_pc_stability(model_type='sae', dataset='mnist', compare_final_epoch=False, size_ls=None, num_models=20, num_epochs=60)
    # analyze_pc_stability(model_type='dae', dataset='mnist', compare_final_epoch=False, size_ls=mnist_size_ls, num_models=20, num_epochs=60)
    # analyze_pc_stability(model_type='sae', dataset='cifar', compare_final_epoch=False, size_ls=None, num_models=5, num_epochs=60)
    # analyze_pc_stability(model_type='dae', dataset='cifar', compare_final_epoch=False, size_ls=cifar_size_ls, num_models=5, num_epochs=60)

    create_combined_stability_heatmaps(dataset='mnist', compare_final_epoch=True)
    # create_combined_stability_heatmaps(dataset='cifar', compare_final_epoch=True)

    create_combined_stability_heatmaps(dataset='mnist', compare_final_epoch=False)
    # create_combined_stability_heatmaps(dataset='cifar', compare_final_epoch=False)