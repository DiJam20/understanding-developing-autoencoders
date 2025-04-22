import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch

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


def encode_dataset(model, dataset_images, dataset="mnist"):
    """
    Encode an entire dataset using the provided model.
    
    Args:
        model: Trained autoencoder model
        dataset_images: Images from the dataset
        dataset: Dataset name ('mnist' or 'cifar')
        
    Returns:
        encodings: Encoded representations
        labels: Corresponding labels
    """
    # Convert to torch tensor if needed
    if isinstance(dataset_images, np.ndarray):
        dataset_images = torch.tensor(dataset_images, dtype=torch.float32)
    
    with torch.no_grad():
        if dataset.lower() == "mnist":
            encoded = model.encode(dataset_images.view(dataset_images.size(0), -1))
        else:
            encoded = model.encode(dataset_images)
    
    return encoded.cpu().numpy()


def get_neuron_importance(classifier):
    """
    Extract and process coefficients from the logistic regression model to determine neuron importance.
    
    Args:
        classifier: Trained LogisticRegression model
        
    Returns:
        importance: Average absolute importance per neuron
    """
    # Extract coefficients
    coeffs = classifier.coef_
    
    # Take the absolute value and average across classes
    importance = np.mean(np.abs(coeffs), axis=0)
    
    return importance


def get_neuron_group_importance(importance, neuron_groups):
    """
    Calculate average importance for each neuron group.
    
    Args:
        importance: Array of neuron importance values
        neuron_groups: List of indices defining the end of each group
        
    Returns:
        List of average importance values for each group
    """
    group_importance = []
    start_indices = [0] + [neuron_groups[i-1] for i in range(1, len(neuron_groups))]
    
    for start_idx, end_idx in zip(start_indices, neuron_groups):
        group_avg = np.mean(importance[start_idx:end_idx])
        group_importance.append(group_avg)
    
    return group_importance


def analyze_model_features(model_index, dataset, base_path, images, labels):
    """
    Analyze neuron importance for a specific model.
    
    Args:
        model_index: Index of the model to analyze
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    
    if dataset.lower() == "mnist":
        model_path = f'{base_path}mnist_models/'
        sae = load_model(f"{model_path}sae/{model_index}", 'sae', 59)
        dae = load_model(f"{model_path}dae/{model_index}", 'dae', 59)
    else:
        model_path = f"{base_path}cifar_models/"
        sae = load_conv_model(f"{model_path}sae/{model_index}", 'sae', 59)
        dae = load_conv_model(f"{model_path}dae/{model_index}", 'dae', 59, size_ls=[128]*60)
        
    sae_train_encodings = encode_dataset(sae, images, dataset)
    dae_train_encodings = encode_dataset(dae, images, dataset)
    
    sae_classifier = LogisticRegression(max_iter=3000)
    dae_classifier = LogisticRegression(max_iter=3000)
    
    sae_classifier.fit(sae_train_encodings, labels)
    dae_classifier.fit(dae_train_encodings, labels)
    
    # Extract neuron importance
    sae_importance = get_neuron_importance(sae_classifier)
    dae_importance = get_neuron_importance(dae_classifier)
    
    return sae_importance, dae_importance


def plot_grouped_importance(sae_importance, dae_importance, neuron_groups, dataset="mnist"):
    """
    Plot neuron importance grouped by neuron groups, side by side for SAE and DAE.
    
    Args:
        sae_importance: SAE neuron importance values
        dae_importance: DAE neuron importance values
        neuron_groups: List of indices defining the end of each group
        dataset: Dataset name ('mnist' or 'cifar')
    """
    sae_group_importance = get_neuron_group_importance(sae_importance, neuron_groups)
    dae_group_importance = get_neuron_group_importance(dae_importance, neuron_groups)
    
    # Create labels for each neuron group
    start_indices = [1] + [neuron_groups[i-1] + 1 for i in range(1, len(neuron_groups))]
    x_labels = [f"{start}-{end}" for start, end in zip(start_indices, neuron_groups)]
    
    plt.rc('font', size=28)
    x_indices = np.arange(len(neuron_groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    # Plot SAE
    sae_bars = ax1.bar(x_indices, sae_group_importance, color='#1a7adb')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Neuron Classification Influence')
    ax1.set_title('AE')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot DAE
    dae_bars = ax2.bar(x_indices, dae_group_importance, color='#e82817')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')
    ax2.set_title('DevAE')
    
    max_val = max(max(sae_group_importance), max(dae_group_importance))
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    fig.suptitle('Neuron Group Importance')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(f"Results/figures/png/{dataset}_grouped_neuron_importance", dpi=300, bbox_inches='tight')
    plt.savefig(f"Results/figures/svg/{dataset}_grouped_neuron_importance", bbox_inches='tight')
    plt.close()


def compute_neuron_importance(num_models, dataset="mnist", base_path="/home/david/", neuron_groups=None):
    """
    Compute average neuron importance across multiple model iterations and plot grouped importance.
    
    Args:
        num_models: Number of models to evaluate
        dataset: Dataset name ('mnist' or 'cifar')
        base_path: Base path to the model directory
    """
    # Results file for average
    result_file = f"Results/{dataset}_neuron_importance.npy"
    
    # Check if average results already exist
    if os.path.exists(result_file):
        print(f"Loading existing average results from {result_file}")
        avg_results = np.load(result_file, allow_pickle=True).item()
    else:
        all_sae_importance = []
        all_dae_importance = []
        all_sae_group_importance = []
        all_dae_group_importance = []

        images, labels = load_cifar_list() if dataset.lower() == "cifar" else load_mnist_list()
        
        # Find importance for each model
        for i in tqdm(range(num_models), desc=f"Processing models"):
            sae_importance, dae_importance = analyze_model_features(
                i, dataset, base_path, images, labels
            )
                
            all_sae_importance.append(sae_importance)
            all_dae_importance.append(dae_importance)
            
            # Calculate group importance for each model
            if neuron_groups is not None:
                sae_group_imp = get_neuron_group_importance(sae_importance, neuron_groups)
                dae_group_imp = get_neuron_group_importance(dae_importance, neuron_groups)
                all_sae_group_importance.append(sae_group_imp)
                all_dae_group_importance.append(dae_group_imp)
        
        avg_sae_importance = np.mean(all_sae_importance, axis=0)
        avg_dae_importance = np.mean(all_dae_importance, axis=0)
        
        avg_results = {
            'sae_importance': avg_sae_importance,
            'dae_importance': avg_dae_importance,
            'neuron_groups': neuron_groups,
            'num_models': num_models
        }
        
        if neuron_groups is not None and len(all_sae_group_importance) > 0:
            avg_results['all_sae_group_importance'] = all_sae_group_importance
            avg_results['all_dae_group_importance'] = all_dae_group_importance
        
        np.save(result_file, avg_results)

    return avg_results


def plot_grouped_importance(sae_importance, dae_importance, neuron_groups, dataset="mnist", avg_results=None):
    """
    Plot neuron importance grouped by neuron groups, side by side for SAE and DAE with error bars.
    
    Args:
        sae_importance: SAE neuron importance values
        dae_importance: DAE neuron importance values
        neuron_groups: List of indices defining the end of each group
        dataset: Dataset name ('mnist' or 'cifar')
        avg_results: Optional dictionary containing additional data for error bars
    """
    # Calculate group importance
    sae_group_importance = get_neuron_group_importance(sae_importance, neuron_groups)
    dae_group_importance = get_neuron_group_importance(dae_importance, neuron_groups)
    
    sae_group_error = np.zeros_like(sae_group_importance)
    dae_group_error = np.zeros_like(dae_group_importance)
    
    # Compute std
    if avg_results is not None and 'all_sae_group_importance' in avg_results:
        all_sae_group = np.array(avg_results['all_sae_group_importance'])
        all_dae_group = np.array(avg_results['all_dae_group_importance'])
        num_models = len(all_sae_group)
        
        if num_models > 1:
            sae_group_error = np.std(all_sae_group, axis=0)
            dae_group_error = np.std(all_dae_group, axis=0)
    
    start_indices = [1] + [neuron_groups[i-1] + 1 for i in range(1, len(neuron_groups))]
    x_labels = [f"{start}-{end}" for start, end in zip(start_indices, neuron_groups)]
    
    # plt.rc('font', size=28)
    x_indices = np.arange(len(neuron_groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.268*0.49, 2.6), dpi=300)
    
    # Plot SAE
    sae_bars = ax1.bar(x_indices, sae_group_importance, color='#1a7adb', 
                       yerr=sae_group_error, capsize=4, ecolor='black')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_labels, rotation=90)
    # ax1.set_xlabel('Neuron Group')
    ax1.set_ylabel('Classification Contribution')
    # ax1.set_title('AE')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot DAE
    dae_bars = ax2.bar(x_indices, dae_group_importance, color='#e82817',
                       yerr=dae_group_error, capsize=4, ecolor='black')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_labels, rotation=90)
    # ax2.set_xlabel('Neuron Group')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_ylabel('')
    # ax2.set_title('DevAE')

    ax1.legend([sae_bars, dae_bars], ['AE', 'Dev-AE'], loc='upper left')
    
    max_val = max(
        max(np.array(sae_group_importance) + np.array(sae_group_error)), 
        max(np.array(dae_group_importance) + np.array(dae_group_error))
    )
    ax1.set_ylim(0, max_val * 1.1)
    ax2.set_ylim(0, max_val * 1.1)
    
    fig.supxlabel('Neuron Group', x=0.57, y=0.1, fontsize=11)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    
    plt.savefig(f"Results/figures/png/{dataset}_grouped_neuron_importance.png", dpi=300)
    plt.savefig(f"Results/figures/svg/{dataset}_grouped_neuron_importance.svg")
    plt.savefig(f"Results/figures/pdf/{dataset}_grouped_neuron_importance.pdf")
    plt.close()


def run_classification_importance_analysis(num_models, base_path="/home/david/", dataset="mnist", size_ls=None):
    """
    Run analyses.
    
    Args:
        num_models: Number of models to evaluate
        base_path: Base path to the model directory
        dataset: Dataset name ('mnist or 'cifar')
        size_ls: List of bottleneck sizes
    """

    print(f"Running analysis for {dataset}...")
    neuron_groups = sorted(set(size_ls))
    avg_results = compute_neuron_importance(num_models, dataset, base_path, neuron_groups)
    
    plot_grouped_importance(
        avg_results['sae_importance'], 
        avg_results['dae_importance'], 
        avg_results['neuron_groups'], 
        dataset,
        avg_results
    )

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

    run_classification_importance_analysis(40, base_path="/home/david/", dataset="mnist", size_ls=mnist_size_ls)
    run_classification_importance_analysis(10, base_path="/home/david/", dataset="cifar", size_ls=cifar_size_ls)