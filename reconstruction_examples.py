import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

def plot_reconstructions(dataset="mnist", base_path="/home/david/", iteration=0, num_examples=5, save_path="Results/figures/"):
    """
    Plot original images and their reconstructions using SAE and DAE models.
    
    Args:
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
        iteration: Model iteration/index to use
        num_examples: Number of examples to show
        save_path: Base path to save the figures
    """    
    # Load test images
    if dataset.lower() == "mnist":
        test_images, _ = load_mnist_list()
        images = test_images[:num_examples]
        model_path = f'{base_path}mnist_models/'
        sae = load_model(f"{model_path}sae/{iteration}", 'sae', 59)
        dae = load_model(f"{model_path}dae/{iteration}", 'dae', 59)
        images_tensor = torch.tensor(images, dtype=torch.float32)
        with torch.no_grad():
            flattened = images_tensor.view(images_tensor.size(0), -1)
            _, sae_decoded = sae(flattened)
            _, dae_decoded = dae(flattened)
            sae_reconstructions = sae_decoded.view(num_examples, 28, 28).cpu().numpy()
            dae_reconstructions = dae_decoded.view(num_examples, 28, 28).cpu().numpy()
    else:
        test_images, _ = load_cifar_list()
        images = test_images[:num_examples]
        model_path = f"{base_path}cifar_models/"
        sae = load_conv_model(f"{model_path}sae/{iteration}", 'sae', 59)
        dae = load_conv_model(f"{model_path}dae/{iteration}", 'dae', 59)
        images_tensor = torch.tensor(images, dtype=torch.float32)
        with torch.no_grad():
            _, sae_decoded = sae(images_tensor)
            _, dae_decoded = dae(images_tensor)
            sae_reconstructions = sae_decoded.view(num_examples, 3, 32, 32).cpu().numpy()
            dae_reconstructions = dae_decoded.view(num_examples, 3, 32, 32).cpu().numpy()
        # Transpose images from (N, C, H, W) to (N, H, W, C) for plotting
        images = np.transpose(images, (0, 2, 3, 1))
        sae_reconstructions = np.transpose(sae_reconstructions, (0, 2, 3, 1))
        dae_reconstructions = np.transpose(dae_reconstructions, (0, 2, 3, 1))
    
    fig, axes = plt.subplots(3, num_examples, figsize=(6.268*0.35, 2))
    
    row_labels = ["Original", "AE", "Dev-AE"]
    plt.rc('font', size=11)
    for row in range(3):
        for col in range(num_examples):
            # Remove all labels
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
            
            if row == 0:  # Original
                img_to_show = images[col]
            elif row == 1:  # SAE
                img_to_show = sae_reconstructions[col]
            else:  # DAE
                img_to_show = dae_reconstructions[col]
            
            if dataset.lower() == "mnist":
                axes[row, col].imshow(img_to_show, cmap='gray')
            else:
                img_to_show = np.clip(img_to_show, 0, 1)
                axes[row, col].imshow(img_to_show)        
        
        # Set row label
        axes[row, 0].set_ylabel(row_labels[row], rotation=90, fontsize=11, labelpad=15, va="center")
    
    # plt.suptitle(f"{dataset.upper()} Image Reconstructions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_path}png/{dataset}_reconstructions.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}svg/{dataset}_reconstructions.svg", bbox_inches='tight')
    plt.savefig(f"{save_path}eps/{dataset}_reconstructions.eps", bbox_inches='tight')
    plt.savefig(f"{save_path}pdf/{dataset}_reconstructions.pdf", bbox_inches='tight')
    plt.close()


def create_reconstruction_examples(base_path="/home/david/", iteration=0, num_examples=5):
    """
    Generate reconstruction examples for MNIST and CIFAR datasets.
    
    Args:
        base_path: Base path to the model directory
        iteration: Model iteration/index to use
        num_examples: Number of examples to show
    """
    print("Generating MNIST reconstruction examples...")
    plot_reconstructions("mnist", base_path, iteration, num_examples)
    
    print("Generating CIFAR reconstruction examples...")
    plot_reconstructions("cifar", base_path, iteration, num_examples)

if __name__ == "__main__":
    create_reconstruction_examples(num_examples=3)