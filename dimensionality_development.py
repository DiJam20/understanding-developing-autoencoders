import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tqdm import tqdm
from model_utils import *

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11
})


#Author: Francesco Mottes
#Date  : 15-Oct-2019
#-----------------------------
def twonn_dimension(data, return_xy=False):
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    -----------
    Parameters:
    data : 2d array-like
        2d data matrix. Samples on rows and features on columns.
    return_xy : bool (default=False)
        Whether to return also the coordinate vectors used for the linear fit.
    -----------
    Returns:
    d : int
        Intrinsic dimension of the dataset according to TWO-NN.
    x : 1d array (optional)
        Array with the -log(mu) values.
    y : 1d array (optional)
        Array with the -log(F(mu_{sigma(i)})) values.
    -----------
    References:
    [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information
        (https://doi.org/10.1038/s41598-017-11873-y)
    """
    data = np.array(data)
    N = len(data)
    
    #mu = r2/r1 for each data point
    mu = []
    for i,x in enumerate(data):
        
        dist = np.sort(np.sqrt(np.sum((x-data)**2, axis=1)))
        r1, r2 = dist[dist>0][:2]

        mu.append((i+1,r2/r1))
        

    #permutation function
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))

    mu = dict(mu)

    #cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N

    #fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])

    #avoid having log(0)
    x = x[y>0]
    y = y[y>0]

    y = -1*np.log(y)

    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
        
    if return_xy:
        return d, x, y
    else: 
        return d
    

def participation_ratio(activation_matrix):
    # Calculate covariance matrix and participation ratio directly on bottleneck activations
    cov_matrix = np.cov(activation_matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.abs(eigenvalues)
    
    # Apply participation ratio formula to eigenvalues
    if np.sum(eigenvalues) > 0:
        dimensionality = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
    else:
        dimensionality = 0
        
    return dimensionality


def calculate_dimensionality_for_single_model(model_idx, model_type, test_images, dataset, size_ls, num_epochs, base_path):
    if dataset.lower() == 'mnist':
        base_path = f"{base_path}mnist_models/{model_type}/{model_idx}"
        loader_function = load_model
    elif dataset.lower() == 'cifar':
        base_path = f"{base_path}cifar_models/{model_type}/{model_idx}"
        loader_function = load_conv_model
    
    dimensionality_over_time = []
    
    # Calculate dimensionality for each epoch
    for epoch in tqdm(range(num_epochs), desc=f"Model {model_idx} epochs", leave=False):
        # Load model for this epoch
        ae = loader_function(base_path, model_type=model_type, epoch=epoch, size_ls=size_ls)
        
        # Get bottleneck activations for all test images
        all_activations = []
        for image in test_images:
            if dataset.lower() == 'mnist':
                image = torch.tensor(image, dtype=torch.float32).reshape(-1)
                with torch.no_grad():
                    encoded = ae.encode(image)
                    all_activations.append(encoded.detach().numpy())
            else:
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    encoded = ae.encode(image)
                    encoded = torch.squeeze(encoded)
                    all_activations.append(encoded.detach().numpy())
        
        # Stack all activations into a single matrix [n_samples, n_features]
        activations_matrix = np.stack(all_activations)

        # Calculate dimensionality using twonn
        twonn_dimensionality = twonn_dimension(activations_matrix)
        dimensionality_over_time.append(twonn_dimensionality)
    
    return dimensionality_over_time


def compute_dimensionality_matrix(model_type, dataset='mnist', size_ls=None, num_models=10, num_epochs=60, base_path='/home/david/'):
    result_file = f"Results/{dataset}_{model_type}_dimensionality.npy"

    # Check if results already exist to avoid recomputation
    if os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        return None
    
    if dataset.lower() == 'mnist':
        test_images, _ = load_mnist_list()
    elif dataset.lower() == 'cifar':
        test_images, _ = load_cifar_list()
    
    all_angles = []
    for idx in range(num_models):
        result = calculate_dimensionality_for_single_model(idx, model_type, test_images, dataset, size_ls, num_epochs, base_path)
        all_angles.append(result)

    np.save(result_file, all_angles)
    print(f"Results saved to {result_file}")


def plot_dimensionality_comparison(dataset):
    sae_file = f"Results/{dataset}_sae_dimensionality.npy"
    dae_file = f"Results/{dataset}_dae_dimensionality.npy"
    if not os.path.exists(sae_file) or not os.path.exists(dae_file):
        print(f"Results files not found.")
        return

    sae_results = np.load(sae_file, allow_pickle=True)
    dae_results = np.load(dae_file, allow_pickle=True)

    # Calculate mean and standard deviation across models
    sae_mean = np.mean(sae_results, axis=0)
    sae_std = np.std(sae_results, axis=0)
    dae_mean = np.mean(dae_results, axis=0)
    dae_std = np.std(dae_results, axis=0)

    plt.figure(figsize=(6.268*0.45, 2.2), dpi=300)
    epochs = np.arange(1, len(sae_mean) + 1)

    plt.plot(epochs, sae_mean, label='AE', color='#1a7adb', linewidth=2)
    plt.fill_between(epochs, sae_mean - sae_std, sae_mean + sae_std, color='#1a7adb', alpha=0.2)
    plt.plot(epochs, dae_mean, label='Dev-AE', color='#e82817', linewidth=2)
    plt.fill_between(epochs, dae_mean - dae_std, dae_mean + dae_std, color='#e82817', alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Intrinsic Dimensionality')
    # plt.title(f'Dimensionality ({dataset.upper()})', fontsize=20, pad=20)
    plt.xticks([0, 29, 59], [1, 30, 60])
    plt.legend(loc='lower right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    
    plt.savefig(f"Results/figures/png/{dataset}_dimensionality_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_dimensionality_comparison.svg", bbox_inches='tight')
    plt.savefig(f"Results/figures/eps/{dataset}_dimensionality_comparison.eps", bbox_inches='tight')
    plt.savefig(f"Results/figures/pdf/{dataset}_dimensionality_comparison.pdf", bbox_inches='tight')
    plt.close()


def run_dimensionality_analysis(dataset, size_ls, num_models, num_epochs, base_path):
    print(f"Running dimensionality analysis for SAE on {dataset}...")
    compute_dimensionality_matrix('sae', dataset, size_ls, num_models, num_epochs, base_path)
    
    print(f"Running dimensionality analysis for DAE on {dataset}...")
    compute_dimensionality_matrix('dae', dataset, size_ls, num_models, num_epochs, base_path)

    plot_dimensionality_comparison(dataset)
    
    print(f"Dimensionality analysis for {dataset} complete.")

if __name__ == "__main__":
    base_path = '/home/david/'

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
    
    run_dimensionality_analysis('mnist', mnist_size_ls, num_models=40, num_epochs=60, base_path=base_path)
    run_dimensionality_analysis('cifar', cifar_size_ls, num_models=10, num_epochs=60, base_path=base_path)