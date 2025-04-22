import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
from solver import *
from model_utils import *

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 11
})


def get_train_loss_per_epoch(model_type, dataset, num_models = 10, base_path = '/home/david/'):
    model_losses = []
    
    for model_idx in range(num_models):
        file_path = f'{base_path}{dataset}_models/{model_type}/{model_idx}/'
        if dataset == 'mnist':
            train_losses = np.load(file_path + 'all_train_losses.npy')
        elif dataset == 'cifar':
            train_losses = np.load(file_path + 'train_loss.npy')
        if dataset == 'mnist':
            train_losses = np.mean(train_losses, axis=1)

        # Convert MSE to RMSE
        rmse_normalized = np.sqrt(train_losses)
        # Denormalize RMSE
        rmse_original = rmse_normalized * 0.3081
        # Convert back to MSE
        train_losses = rmse_original ** 2

        model_losses.append(train_losses)
    
    model_losses_array = np.array(model_losses)

    return model_losses_array


def get_vali_loss_per_epoch_mnist(model_type, dataset, num_models=10, base_path='/home/david/'):
    # Check if the file already exists
    result_file = f'Results/mnist_{model_type}_vali_loss.npy'
    import os
    if os.path.exists(result_file):
        print(f"Loading cached validation losses from {result_file}")
        return np.load(result_file)
    
    print(f"Computing validation losses for {model_type} on {dataset}...")
    model_losses = []

    _, _, test_loader = load_mnist()
    
    # Loop through each model with tqdm progress bar
    for model_idx in tqdm(range(num_models), desc="Processing models", leave=True):
        # Store losses for each epoch of the current model
        epoch_losses = []

        # Loop through each epoch with nested tqdm progress bar
        for epoch in tqdm(range(60), desc=f"Model {model_idx} epochs", leave=False):
            file_path = f'{base_path}{dataset}_models/{model_type}/{model_idx}/'
            ae = load_model(file_path, model_type, epoch)
            test_loss, _, _ = test(ae, test_loader, device='cpu')
            epoch_losses.append(test_loss)

        model_losses.append(epoch_losses)
    
    model_losses_array = np.array(model_losses)
    
    # Apply the transformations to the array directly
    # Convert MSE to RMSE
    rmse_normalized = np.sqrt(model_losses_array)
    # Denormalize RMSE
    rmse_original = rmse_normalized * 0.3081
    # Convert back to MSE
    model_losses_array = rmse_original ** 2
    
    print(f"Saving validation losses to {result_file}")
    np.save(result_file, model_losses_array)
    
    return model_losses_array


def get_vali_loss_per_epoch(model_type, dataset, num_models = 10, base_path = '/home/david/'):
    model_losses = []
    
    for model_idx in range(num_models):
        file_path = f'{base_path}{dataset}_models/{model_type}/{model_idx}/'
        train_losses = np.load(file_path + 'vali_loss.npy')

        # Convert MSE to RMSE
        rmse_normalized = np.sqrt(train_losses)
        # Denormalize RMSE
        rmse_original = rmse_normalized * 0.3081
        # Convert back to MSE
        train_losses = rmse_original ** 2

        model_losses.append(train_losses)
    
    model_losses_array = np.array(model_losses)

    return model_losses_array


def plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, sae_vali_loss, dae_vali_loss, dataset):
    sae_train_mean = np.mean(sae_train_loss, axis=0)
    sae_train_std = np.std(sae_train_loss, axis=0)
    dae_train_mean = np.mean(dae_train_loss, axis=0)
    dae_train_std = np.std(dae_train_loss, axis=0)

    sae_vali_mean = np.mean(sae_vali_loss, axis=0)
    sae_vali_std = np.std(sae_vali_loss, axis=0)
    dae_vali_mean = np.mean(dae_vali_loss, axis=0)
    dae_vali_std = np.std(dae_vali_loss, axis=0)

    plt.figure(figsize=(6.266*0.47, 2.5), dpi=300)
    
    sae_train, = plt.plot(sae_train_mean, label='AE Train Loss', color='#1a7adb', linewidth=2)
    plt.fill_between(range(len(sae_train_loss[0])), 
                     sae_train_mean - sae_train_std, 
                     sae_train_mean + sae_train_std, 
                     color='#1a7adb', alpha=0.2)
    
    dae_train, = plt.plot(dae_train_mean, label='Dev-AE Train Loss', color='#e82817', linewidth=2)
    plt.fill_between(range(len(sae_train_loss[0])), 
                     dae_train_mean - dae_train_std, 
                     dae_train_mean + dae_train_std, 
                     color='#e82817', alpha=0.2)
    
    sae_vali, = plt.plot(sae_vali_mean, label='AE Vali Loss', color='#1a7adb', linewidth=2, linestyle='--')
    plt.fill_between(range(len(sae_vali_loss[0])), 
                    sae_vali_mean - sae_vali_std, 
                    sae_vali_mean + sae_vali_std, 
                    color='#1a7adb', alpha=0.2)
    
    dae_vali, = plt.plot(dae_vali_mean, label='Dev-AE Vali Loss', color='#e82817', linewidth=2, linestyle='--')
    plt.fill_between(range(len(sae_vali_loss[0])), 
                    dae_vali_mean - dae_vali_std, 
                    dae_vali_mean + dae_vali_std, 
                    color='#e82817', alpha=0.2)
        
    plt.xticks([0, 29, 59], [1, 30, 60])
    plt.ylim(0.005, 0.105)
    plt.yticks([0.01, 0.1])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    # plt.title('Loss Curves', pad=20)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    first_legend = plt.legend([sae_train, dae_train], ['AE', 'Dev-AE'], 
                             bbox_to_anchor=(1.0, 1.1),
                             loc='upper right', 
                             title='Train Loss',
                             title_fontsize='11',
                             fontsize='11')
    ax.add_artist(first_legend)
    
    plt.legend([sae_vali, dae_vali], ['AE', 'Dev-AE'], 
                bbox_to_anchor=(1.0, 0.68),
                loc='upper right', 
                title='Validation Loss',
                title_fontsize='11',
                fontsize='11')

    plt.tight_layout()

    plt.savefig(f'Results/figures/png/{dataset}_accuracy_over_epochs.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'Results/figures/svg/{dataset}_accuracy_over_epochs.svg', bbox_inches='tight')
    plt.savefig(f'Results/figures/eps/{dataset}_accuracy_over_epochs.eps', bbox_inches='tight')
    plt.savefig(f'Results/figures/pdf/{dataset}_accuracy_over_epochs.pdf', bbox_inches='tight')
    plt.close()


def create_plots():
    sae_train_loss = get_train_loss_per_epoch('sae', 'mnist')
    dae_train_loss = get_train_loss_per_epoch('dae', 'mnist')
    sae_vali_loss = get_vali_loss_per_epoch_mnist('sae', 'mnist')
    dae_vali_loss = get_vali_loss_per_epoch_mnist('dae', 'mnist')
    plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, sae_vali_loss, dae_vali_loss, 'mnist')

    sae_train_loss = get_train_loss_per_epoch('sae', 'cifar')
    dae_train_loss = get_train_loss_per_epoch('dae', 'cifar')
    sae_vali_loss = get_vali_loss_per_epoch('sae', 'cifar')
    dae_vali_loss = get_vali_loss_per_epoch('dae', 'cifar')
    plot_accuracy_over_epochs(sae_train_loss, dae_train_loss, sae_vali_loss, dae_vali_loss, 'cifar')

if __name__ == "__main__":
    create_plots()