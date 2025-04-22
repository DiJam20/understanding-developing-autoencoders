import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from autoencoder import NonLinearAutoencoder, ConvAutoencoder

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(len(mnist_train) * 0.8)
    validation_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, validation_size])

    batch_size = 128
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=6)
    validation_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False, num_workers=6)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=6)
    return train_loader, validation_loader, test_loader


def load_cifar():
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(len(cifar_train) * 0.8)
    validation_size = len(cifar_train) - train_size
    cifar_train, cifar_val = torch.utils.data.random_split(cifar_train, [train_size, validation_size])

    batch_size = 128
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=6)
    validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False, num_workers=6)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=6)
    return train_loader, validation_loader, test_loader


def load_mnist_tensor():
    _, _, test_loader = load_mnist()

    tensor_test_images = []
    test_labels = torch.zeros(10000, dtype=torch.long)

    current_idx = 0
    for images, labels in test_loader:
        batch_size = images.size(0)
        
        test_labels[current_idx:current_idx + batch_size] = labels
        
        for i in range(batch_size):
            tensor_test_images.append(images[i].flatten())
        
        current_idx += batch_size

    tensor_test_images = torch.stack(tensor_test_images)

    return tensor_test_images, test_labels


def load_cifar_tensor():
    _, _, test_loader = load_cifar()

    tensor_test_images = []
    test_labels = torch.zeros(10000, dtype=torch.long)

    current_idx = 0
    for images, labels in test_loader:
        batch_size = images.size(0)
        
        test_labels[current_idx:current_idx + batch_size] = labels
        
        for i in range(batch_size):
            tensor_test_images.append(images[i].flatten())
        
        current_idx += batch_size

    tensor_test_images = torch.stack(tensor_test_images)

    return tensor_test_images, test_labels


def load_mnist_list():
    """
    Load MNIST test data and normalise it.

    Returns:
        np.array: List of MNIST images
        np.array: List of MNIST labels
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False, num_workers=6)

    test_images = []
    test_labels = []

    for batch_idx, (data, target) in enumerate(test_loader):
        test_images.append(data)
        test_labels.append(target)

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)
    test_images = test_images.squeeze(1)

    return test_images, test_labels


def load_cifar_list():
    """
    Load CIFAR10 test data and normalise it.

    Returns:
        np.array: List of CIFAR10 images
        np.array: List of CIFAR10 labels
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=6)

    test_images = []
    test_labels = []

    for batch_idx, (data, target) in enumerate(test_loader):
        test_images.append(data)
        test_labels.append(target)

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    return test_images, test_labels


def load_model(model_path, model_type, epoch, size_ls=None):
    n_input = 28*28
    n_layers = 3
    sae_n_hidden_ls = [512, 128, 32]

    if size_ls is None:
        size_ls = [4, 4, 4, 4, 4, 10,
                10, 10, 10, 10, 16, 16,
                16, 16, 16, 16, 16, 24,
                24, 24, 24, 24, 24, 24, 
                32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32]
    
    dae_n_hidden_ls = [512, 128, size_ls[epoch]]
    
    if model_type.lower() == 'sae':
        model = NonLinearAutoencoder(n_input, sae_n_hidden_ls, n_layers)
    else:
        model = NonLinearAutoencoder(n_input, dae_n_hidden_ls, n_layers)
    device = torch.device("cpu")
    weights = torch.load(
        f"{model_path}/model_weights_epoch{epoch}.pth", 
        map_location=device, 
        weights_only=True
    )
    model.load_state_dict(weights)
    return model


def load_conv_model(model_path, model_type, epoch, size_ls=None):
    if size_ls is None:
        size_ls = [6, 6, 6, 6, 6, 6, 
                10, 10, 10, 10, 10, 10,
                16, 16, 16, 16, 16, 16,
                28, 28, 28, 28, 28, 28,
                48, 48, 48, 48, 48, 48, 48, 48, 48,
                90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
                128, 128, 128, 128, 128, 128, 128, 128, 
                128, 128, 128, 128, 128, 128, 128, 128, 128
                ]
    
    if model_type == "sae":
        model = ConvAutoencoder(latent_dim=size_ls[-1])
    else:
        model = ConvAutoencoder(latent_dim=size_ls[epoch])
    device = torch.device("cpu")
    weights = torch.load(
        f"{model_path}/model_weights_epoch{epoch}.pth", 
        map_location=device, 
        weights_only=True
        )
    model.load_state_dict(weights)
    return model


def cosine_angle_between_vectors(vec_a, vec_b):
    """
    Compute the cosine angle between two vectors.
    cos(theta) = (vec_a · vec_b) / (||vec_a|| * ||vec_b||)
    
    Args:
        vec_a: First vector
        vec_b: Second vector
    
    Returns:
        angle: Angle between the two vectors in degrees, normalized to be between 0 and 90 degrees
    """
    # check if vectors have more than one dimension
    if vec_a.ndim > 1:
        vec_a = vec_a.flatten()
    if vec_b.ndim > 1:
        vec_b = vec_b.flatten()
    numerator = np.dot(vec_a, vec_b)
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return np.nan
    cos_value = np.clip(numerator / denominator, -1.0, 1.0)
    angle = np.arccos(cos_value) * 180 / np.pi
    return min(angle, 180 - angle)

# Source: https://fairyonice.github.io/Low-and-High-pass-filtering-experiments.html
# Yumi Kondo, 22.09.2018
def draw_circle(shape, diameter):
    assert len(shape) == 2
    circle_mask = np.zeros(shape,dtype=np.bool)
    center = np.array(circle_mask.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            circle_mask[iy,ix] = (iy - center[0])**2 + (ix - center[1])**2 < diameter**2
    return circle_mask


# Source: https://fairyonice.github.io/Low-and-High-pass-filtering-experiments.html
# Yumi Kondo, 22.09.2018
def filter_circle(circle_mask, full_fft_filter):
    temp = np.zeros(full_fft_filter.shape,dtype=complex)
    temp[circle_mask] = full_fft_filter[circle_mask]
    return(temp)


def add_frequency_noise(image, noise_level=1, inner_diameter=3, outer_diameter=7):
    # Ensure image is 2D
    if len(image.shape) > 2:
        if image.shape[0] == 1:  # Single channel
            image = image.squeeze(0)
        elif image.shape[0] == 3:  # RGB
            # Use grayscale conversion if RGB
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    
    circle_mask = draw_circle(shape=image.shape,diameter=inner_diameter)
    outer_circle_mask = draw_circle(shape=image.shape, diameter=outer_diameter)
    ring_mask = outer_circle_mask & ~circle_mask

    noise = np.random.randn(*image.shape)
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    noise_fft_ring = noise_fft * ring_mask * noise_level
    frequency_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_ring))
    frequency_noise = np.real(frequency_noise)

    # Normalize noise to match the magnitude to the noise level
    noise_magnitude = np.sqrt(np.mean(frequency_noise**2))
    noise_scale = noise_level / noise_magnitude
    frequency_noise = frequency_noise * noise_scale

    img = image + frequency_noise

    return img, noise_scale