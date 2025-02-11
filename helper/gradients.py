import torch
import numpy as np

from tqdm import tqdm
from helper.global_variables import DEVICE
from torch.utils.data import Dataset

# TODO Add documentation


class ModifiedDataset(Dataset):
    def __init__(self, modified_images, original_dataset):
        self.images = modified_images
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        _, label = self.original_dataset[idx]
        return self.images[idx], label


def __get_grad(model, 
               images, 
               indices, 
               dimensions):
    """
    Calculate gradients for one-dimensional activations
    """
    images.requires_grad_(True)
    output = model(images)
    if dimensions == 2:
        target_outputs = torch.stack([output[:, idx[0]] for idx in indices])
    elif dimensions == 3:
        target_outputs = torch.stack([output[:, idx[0], idx[1]] for idx in indices])
    elif dimensions == 4:
        target_outputs = torch.stack([output[:, idx[0], idx[1], idx[2]] for idx in indices])
    else:
        raise ValueError(f"Unsupported dimensions: {dimensions}")
    scalar_output = target_outputs.sum()
    
    # Compute gradient of scalar
    grads = torch.autograd.grad(
        scalar_output,
        images, 
        retain_graph=True)[0]

    return grads


def get_top_k_sus_grads(model, sus_vals, images, k, verbose=False):
    """
    Berechnet Gradienten für die k verdächtigsten Neuronen
    """

    # Finde Top-k Werte und ihre Indizes
    flattened_sus_values = sus_vals.flatten()
    _, top_k_idxs = torch.topk(flattened_sus_values, k)

    # Test-Forward-Pass
    with torch.no_grad():
        single_image = images[0].unsqueeze(0)  # Shape: [1, C, H, W]
        test_output = model(single_image)

    # Index-Konvertierung basierend auf Dimensionalität
    if len(sus_vals.shape) == 1:
        indices = [(idx.item(),) for idx in top_k_idxs]
    else:
        # Konvertiere Indizes und überprüfe Dimensionen
        unraveled = np.unravel_index(top_k_idxs.cpu().numpy(), sus_vals.shape)
        indices = list(zip(*unraveled))  # Konvertiere zu Liste von Tupeln

    if verbose:
        print(f"Output shape: {test_output.shape}")
        print(f"Sus values shape: {sus_vals.shape}")
        print(f"topkidx: {top_k_idxs}")
        print(f"Final indices format: {indices[:5]}")  # Zeige erste 5 Indizes

    dim = test_output.dim()
    return __get_grad(model,
                      images,
                      indices,
                      dim)


def perform_gradient_descent(model,
                             dataset_loader,
                             sus_values,
                             num_neurons,
                             grad_factor,
                             num_iterations=5):
    """
    Modify images using gradient descent method
    """

    modified_images = []
    model.eval()

    for images, _ in tqdm(dataset_loader, desc="Evaluating Images ..."):
        images = images.to(DEVICE).float()  # Konvertiere zu float
        images_altered = images.clone()

        # Iterative Gradientenberechnung
        for _ in range(num_iterations):
            grads = get_top_k_sus_grads(
                model, sus_values, images_altered, num_neurons)

            # print(f"Gradient magnitude: {torch.norm(grads).item():.4f}")
            with torch.no_grad():
                images_altered = images_altered + grad_factor * grads

        modified_images.append(images_altered.cpu().detach())

    modified_images_tensor = torch.cat(modified_images)

    return ModifiedDataset(modified_images_tensor, dataset_loader.dataset)
