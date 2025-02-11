"""Helper functions for visualization and plotting using matplotlib.

This module provides utility functions for visualizing datasets and creating plots.
It includes functions for displaying images from PyTorch datasets and creating
horizontal bar charts.

Functions:
    plot_images: Displays multiple images from a dataset in a row
    show_horizontal_bar_diagram: Creates and displays a horizontal bar chart
"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(image_ds: torch.utils.data.Dataset, 
                num_images: int = 4, 
                output_path: Optional[str] = None,
                print_images: bool=False) -> None:
    """
    Displays the first num_images from a dataset in a row.

    This function creates a matplotlib figure with subplots showing multiple images
    from the provided dataset. It handles both tensor and numpy array inputs,
    automatically converting between formats as needed.

    Args:
        image_ds: PyTorch Dataset or DataLoader containing images
        num_images: Number of images to display. Defaults to 4.

    Returns:
        None. Displays the plot using matplotlib.

    Example:
        >>> dataset = torchvision.datasets.CIFAR10(root='./data', train=True)
        >>> plot_images(dataset, num_images=5)

    Notes:
        - Automatically handles channel dimension ordering (CHW -> HWC)
        - Normalizes pixel values to [0,1] range if needed
        - Works with both RGB and grayscale images
        - Images are displayed without axes for cleaner visualization
    """

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    # Make axes iterable if single image
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        img, _ = image_ds[i]

        # Convert tensor to numpy
        if torch.is_tensor(img):
            img = img.cpu().numpy()

        # Move channels last (CHW -> HWC)
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # Denormalize if in [-1,1] range
        if img.min() < 0:
            img = (img + 1) / 2
        # Normalize if above 1
        elif img.max() > 1:
            img = img / 255.0

        # Ensure proper range
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    if print_images:
        plt.show()
    
    # Save if output path provided
    if output_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    

def show_bar_diagram(labels: list[str], values: list[float]) -> None:
    """
    Create and display a horizontal bar diagram using matplotlib.

    Args:
        labels (list[str]): The labels/categories to be displayed on the y-axis
        values (list[float]): The corresponding values for each label to be displayed as horizontal bars

    Returns:
        None: The function displays the plot but does not return any value

    Example:
        >>> labels = ['A', 'B', 'C']
        >>> values = [1.2, 3.4, 2.1]
        >>> show_horizontal_bar_diagram(labels, values)
    """
    # Create the horizontal bar diagram
    plt.bar(labels, values)

    # Add title and labels
    plt.title('Horizontal Bar Diagram')
    plt.xlabel('Values')
    plt.ylabel('Categories')

    # Show the plot
    plt.show()
