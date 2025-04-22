import os
from pc_stability import analyze_pc_stability
from rf_computation import compute_rfs
from rf_stability import analyze_rf_stability
from rf_specificity import compute_rf_specificity
from hidden_layer_statistics import compute_hidden_layer_activation
from plot_hidden_layer_statistics import *
from power_spectra import *
from pc_noise import *
from bottleneck_activation import run_bottleneck_activation_analysis
from encoding_noise import run_encoding_noise_analysis
from importance_for_classification import run_classification_importance_analysis
from dimensionality_development import run_dimensionality_analysis
from frequency_noise import run_all_frequency_analyses
from reconstruction_examples import create_reconstruction_examples

def main():
    base_path = '/home/david/'

    os.makedirs("Results/figures/png", exist_ok=True)
    os.makedirs("Results/figures/svg", exist_ok=True)
    os.makedirs("Results/figures/eps", exist_ok=True)
    os.makedirs("Results/figures/pdf", exist_ok=True)

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
    
    analyze_pc_stability("sae", dataset='mnist', compare_final_epoch=True, size_ls=None, num_models=10, num_epochs=60)
    analyze_pc_stability("dae", dataset='mnist', compare_final_epoch=True, size_ls=mnist_size_ls, num_models=10, num_epochs=60)
    analyze_pc_stability("sae", dataset='cifar', compare_final_epoch=True, size_ls=None, num_models=1, num_epochs=20)
    analyze_pc_stability("dae", dataset='cifar', compare_final_epoch=True, size_ls=cifar_size_ls, num_models=1, num_epochs=20)

    compute_rfs("sae", dataset='mnist', size_ls=None, num_models=40, num_epochs=60)
    compute_rfs("dae", dataset='mnist', size_ls=mnist_size_ls, num_models=40, num_epochs=60)
    compute_rfs("sae", dataset='cifar', size_ls=None, num_models=10, num_epochs=60)
    compute_rfs("dae", dataset='cifar', size_ls=cifar_size_ls, num_models=10, num_epochs=60)

    analyze_rf_stability("sae", dataset='mnist', compare_final_epoch=True, size_ls=None, num_models=2, num_epochs=60)
    analyze_rf_stability("dae", dataset='mnist', compare_final_epoch=True, size_ls=mnist_size_ls, num_models=2, num_epochs=60)
    analyze_rf_stability("sae", dataset='cifar', compare_final_epoch=True, size_ls=None, num_models=1, num_epochs=60)
    analyze_rf_stability("dae", dataset='cifar', compare_final_epoch=True, size_ls=cifar_size_ls, num_models=2, num_epochs=60)

    compute_rf_specificity("sae", dataset='cifar', num_models=1, size_ls=None, num_epochs=60)
    compute_rf_specificity("dae", dataset='cifar', num_models=1, size_ls=mnist_size_ls, num_epochs=60)

    compute_hidden_layer_activation('sae', 'mnist', num_models=40, num_epochs=1, epoch=59)
    compute_hidden_layer_activation('dae', 'mnist', num_models=40, num_epochs=1, epoch=59)
    compute_hidden_layer_activation('sae', 'cifar', num_models=10, num_epochs=1, epoch=59)
    compute_hidden_layer_activation('dae', 'cifar', num_models=10, num_epochs=1, epoch=59)

    plot_hidden_layer_activation(dataset='mnist')
    plot_hidden_layer_activation(dataset='cifar')
    plot_neuron_activations('mnist')
    plot_neuron_activations('cifar')

    neuron_groups = [6, 10, 16, 28, 90, 128]
    analyze_power_spectra('Results/cifar_sae_rfs.npy', 'Results/cifar_dae_rfs.npy', 2, 59, neuron_groups=neuron_groups)


    run_bottleneck_activation_analysis(40, 'mnist', base_path)
    run_bottleneck_activation_analysis(10, 'cifar', base_path)


    run_encoding_noise_analysis(10, mnist_size_ls, 'mnist', '/home/david/')
    run_encoding_noise_analysis(1, cifar_size_ls, 'cifar', '/home/david/')

    mnist_manipulated_neurons = [(0, 4), (4, 10), (10, 17), (17, 24), (24, 32)]
    cifar_manipulated_neurons = [(0, 6), (6, 10), (10, 16), (16, 28), (28, 48), (48, 90), (90, 128)]

    run_all_pc_analyses(40, "mnist", base_path, manipulated_neurons=mnist_manipulated_neurons)
    run_all_pc_analyses(10, "cifar", base_path, manipulated_neurons=cifar_manipulated_neurons)

    run_dimensionality_analysis('mnist', mnist_size_ls, num_models=2, num_epochs=30, base_path=base_path)
    run_dimensionality_analysis('cifar', cifar_size_ls, num_models=2, num_epochs=60, base_path=base_path)

    run_classification_importance_analysis(40, dataset="mnist", size_ls=mnist_size_ls)
    run_classification_importance_analysis(10, dataset="cifar", size_ls=cifar_size_ls)

    run_all_frequency_analyses(40, 10, base_path)

    create_reconstruction_examples(base_path)

if __name__ == "__main__":
    main()