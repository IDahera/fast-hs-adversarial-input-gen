"""
Helper models and utilities for training, evaluating, and managing neural network models.

This module provides various utilities for handling different types of neural network models,
including model creation, layer configuration, training, and evaluation functionalities.

Classes:
    ModelType (Enum): Enumeration of supported model architectures (DENSE, CONV, MOBILENET, SQUEEZENET)
    LayerConfig (Enum): Enumeration for layer configuration options (EARLY, LATE)

Functions:
    get_model(model_name: str) -> torch.nn.Module:
        Creates and returns a neural network model based on the specified model name.
    
    get_layer_config() -> Dict[str, Dict[int, Callable]]:
        Returns layer extraction configurations for different model architectures.
    
    get_sub_model_config() -> Dict[str, Dict[int, Callable]]:
        Returns sub-model extraction configurations for different model architectures.
    
    get_layer(model_name: str, layer_config: int) -> Callable:
        Returns a function to extract a specific layer from a model.
    
    get_sub_model(model: nn.Module, model_name: str, layer_config: int) -> nn.Module:
        Extracts and returns a sub-model up to a specified layer.
    
    train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5) -> nn.Module:
        Trains a neural network model using specified training data.
    
    evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
        Evaluates model accuracy on test data and returns classification accuracy.
    
    evaluate_model_per_class(model: nn.Module, test_loader: DataLoader, num_classes: int = 10) 
        -> Tuple[Dict[int, int], str]:
        Evaluates model performance per class and returns detailed classification distribution.
"""

import torch
from typing import Dict, Tuple, Callable
from enum import Enum

from collections import Counter
from helper.global_variables import DEVICE
from tqdm import tqdm
from torch import nn, optim
from torchvision import models


# DeepFault Reference ModelS
class DenseModel8x20(nn.Module):
    def __init__(self, input_size=28*28, classes=10):
        super(DenseModel8x20, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),

            # dense with leaky_relu
            nn.Linear(input_size, 20),
            nn.LeakyReLU(),

            # dense_1 with leaky_relu_1
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_2 with leaky_relu_2
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_3 with leaky_relu_3
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_4 with leaky_relu_4
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_5 with leaky_relu_5
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_6 with leaky_relu_6
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_7 with leaky_relu_7
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            # dense_8 with leaky_relu_8
            nn.Linear(20, 20),
            nn.LeakyReLU(),

            nn.Linear(20, classes)
        )

    def forward(self, x):
        return self.layers(x)


class DenseModel10x100(nn.Module):
    def __init__(self, input_size=28*28, classes=10):
        super(DenseModel10x100, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),

            # dense with leaky_relu
            nn.Linear(input_size, 100),
            nn.LeakyReLU(),

            # dense_1 with leaky_relu_1
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_2 with leaky_relu_2
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_3 with leaky_relu_3
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_4 with leaky_relu_4
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_5 with leaky_relu_5
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_6 with leaky_relu_6
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_7 with leaky_relu_7
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_8 with leaky_relu_8
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_9 with leaky_relu_9
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            # dense_10 with leaky_relu_10
            nn.Linear(100, 100),
            nn.LeakyReLU(),

            nn.Linear(100, classes)
        )

    def forward(self, x):
        return self.layers(x)


class DenseModel(nn.Module):
    """A feed-forward neural network with dense (fully connected) layers.
    This model consists of multiple dense layers with ReLU activations,
    designed primarily for image classification tasks.
    Args:
        input_size (int): Size of the input features. Defaults to 784 (28*28)
        classes (int): Number of output classes. Defaults to 10
    Attributes:
        layers (nn.Sequential): Sequential container of the neural network layers
    """

    def __init__(self, input_size=28*28, classes=10):
        super(DenseModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes)
        )

    def forward(self, x):
        return self.layers(x)


# LeNet
class ConvModel(nn.Module):
    """A convolutional neural network based on LeNet architecture.
    This model implements a modified version of LeNet with three convolutional layers
    followed by max pooling and dense layers for classification tasks.
    Args:
        classes (int): Number of output classes. Defaults to 10
    Attributes:
        layers (nn.Sequential): Sequential container of the neural network layers
    """

    def __init__(self, classes=10):
        super(ConvModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, classes)
        )

    def forward(self, x):
        return self.layers(x)


class ModelType(Enum):
    DENSE8X20 = "dense8x20"
    DENSE10X100 = "dense10x100"
    DENSE = "dense"
    CONV = "conv"
    MOBILENET = "mobilenet"
    SQUEEZENET = "squeezenet"


class LayerConfig(Enum):
    EARLY = 1  # Early layer configuration
    LATE = 2   # Late layer configuration
    LAST = 3   # LAST layer configuration


def get_model(model_name: str):
    """
    Retrieve a model based on the provided model name.
    Args:
        model_name (str): The name of the model to retrieve. 
                          Possible values include "dense", "conv", "mobilenet", and "squeezenet".
    Returns:
        torch.nn.Module: The corresponding model instance moved to the specified device.
    Raises:
        Exception: If the provided model name does not match any known models.
    """
    model_mapping = {
        ModelType.DENSE8X20.value:
            lambda: DenseModel8x20(),
        ModelType.DENSE10X100.value:
            lambda: DenseModel10x100(),
        ModelType.DENSE.value:
            lambda: DenseModel(),
        ModelType.CONV.value:
            lambda: ConvModel(),
        ModelType.MOBILENET.value:
            lambda: models.mobilenet_v3_small(pretrained=True),
        ModelType.SQUEEZENET.value:
            lambda: models.squeezenet1_1(
                pretrained=True)
    }

    for model_type in ModelType:
        if model_type.value in model_name:
            model = model_mapping[model_type.value]()
            return model.to(DEVICE)

    raise Exception(f"Invalid Model Name: {model_name}")


def load_model_with_weights(model_path: str) -> nn.Module:
    model_name = model_path.split("/")[-1].split(".")[0]
    
    model = get_model(model_name)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    
    return model


def get_layer_config() -> Dict[str, Dict[int, Callable]]:
    """Define layer extraction functions for different model architectures."""
    return {
        ModelType.DENSE8X20.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[9],
            LayerConfig.LATE.value: lambda m: m.layers[-1]},
        ModelType.DENSE10X100.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[11],
            LayerConfig.LATE.value: lambda m: m.layers[-1]},
        ModelType.CONV.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[3]
        },
        ModelType.DENSE.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[5]
        },
        ModelType.MOBILENET.value: {
            LayerConfig.EARLY.value: lambda m: m.features[0][0],
            LayerConfig.LATE.value: lambda m: m.features[12][0],
            LayerConfig.LAST.value: lambda m: m.features[-1]

        },
        ModelType.SQUEEZENET.value: {
            LayerConfig.EARLY.value: lambda m: m.features[0],
            LayerConfig.LATE.value: lambda m: m.features[12].squeeze,
            LayerConfig.LAST.value: lambda m: m.features[-1]
        }
    }


def get_sub_model_config() -> Dict[str, Dict[int, Callable]]:
    return {
        ModelType.MOBILENET.value: {
            LayerConfig.EARLY.value: lambda m: nn.Sequential(m.features[:1]),
            LayerConfig.LATE.value: lambda m: nn.Sequential(
                *m.features[:12], m.features[12][0]),
            LayerConfig.LAST.value: lambda m: m
        },
        ModelType.SQUEEZENET.value: {
            LayerConfig.EARLY.value: lambda m: nn.Sequential(m.features[:1]),
            LayerConfig.LATE.value: lambda m: nn.Sequential(
                *list(m.features)[:-1], m.features[12].squeeze),
            LayerConfig.LAST.value: lambda m: nn.Sequential(
                *list(m.features)[:-1], m)
        },
        ModelType.DENSE8X20.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[:10],
            LayerConfig.LATE.value: lambda m: m.layers[:-1]
        },
        ModelType.DENSE10X100.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[:12],
            LayerConfig.LATE.value: lambda m: m.layers[:-1]
        },
        ModelType.DENSE.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[:6]
        },
        ModelType.CONV.value: {
            LayerConfig.EARLY.value: lambda m: m.layers[:4]
        }
    }


def get_layer(model_name: str, layer_config: int) -> Callable:
    """Get function to extract specific layer from model.

    Args:
        model_name: Name of model architecture
        layer_config: Layer configuration number
    Returns:
        Function that extracts specified layer
    """
    dataset_prefix = model_name.split("-")[0]
    model_type = model_name.replace(dataset_prefix + "-", "")
    layer_configs = get_layer_config()

    if model_type not in layer_configs or layer_config not in layer_configs[model_type]:
        raise ValueError(f"Invalid model type {
                         model_type} for layer config {layer_config}")

    return layer_configs[model_type][layer_config]


def get_sub_model(model: nn.Module,
                  model_name: str,
                  layer_config: int) -> nn.Module:
    """Extract sub-model up to specified layer.

    Args:
        model: Base model
        model_name: Model architecture name
        layer_config: Layer configuration number
    Returns:
        Sub-model containing layers up to specified point
    """
    sub_model_configs = get_sub_model_config()

    for model_type in ModelType:
        if model_type.value in model_name:
            if layer_config not in sub_model_configs[model_type.value]:
                raise ValueError(f"Invalid layer config {
                                 layer_config} for {model_type.value}")
            return sub_model_configs[model_type.value][layer_config](model)

    raise ValueError(f"Invalid model name: {model_name}")


def get_model_size(model: nn.Module) -> int:
    """Calculate model size in parameters and MB"""
    # Count parameters
    param_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    num_params = sum(p.numel() for p in model.parameters())

    return num_params


# Train and Eval Helpers
def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                epochs: int = 5
                ) -> nn.Module:
    """ Train a PyTorch neural network model.
        This function trains a neural network using CrossEntropyLoss and Adam optimizer
        over the specified number of epochs.
        Args:
            model (nn.Module): The PyTorch model to be trained
            train_loader (torch.utils.data.DataLoader): DataLoader containing the training data
            epochs (int, optional): Number of training epochs. Defaults to 5
        Returns:
            nn.Module: The trained model
        Example:
            >>> model = MyNeuralNet()
            >>> train_loader = DataLoader(dataset, batch_size=32)
            >>> trained_model = train_model(model, train_loader, epochs=10)
        """

    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters())

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, "Train Model ..."):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            outputs = model(images).to(DEVICE)
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    return model


def evaluate_model(model: nn.Module,
                   test_loader: torch.utils.data.DataLoader) -> float:
    """ Evaluates a PyTorch model's accuracy on a test dataset.
        This function computes the classification accuracy of a neural network model
        on a given test dataset using a DataLoader.
        Args:
            model (nn.Module): PyTorch neural network model to evaluate
            test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset
        Returns:
            float: Classification accuracy as a percentage (0-100)
        Example:
            >>> test_accuracy = evaluate_model(my_model, test_dataloader)
            >>> print(f"Test accuracy: {test_accuracy:.2f}%")
    """
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, "Evaluate Model ..."):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def evaluate_model_per_class(model: nn.Module,
                             test_loader: torch.utils.data.DataLoader) -> Dict[int, int]:
    """Evaluates the given model on the test dataset and returns the classification distribution per class.
    Args:
        model (nn.Module): The neural network model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        num_classes (int, optional): The number of classes in the dataset. Default is 10.
    Returns:
        Tuple[Dict[int, int], str]: A tuple containing:
            - A dictionary with the count of predictions for each class.
            - A formatted string log of the classification distribution.
    """

    model.eval()
    predictions_counter = Counter()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, "Evaluate Model ..."):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.cpu().numpy()
            predictions_counter.update(predictions)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return dict(predictions_counter), (100 * correct / total)


def convert_counter_to_str(predictions_counter, num_classes=10):

    # Format output
    total = sum(predictions_counter.values())
    logs = "\nClassification Distribution\n"
    logs += "=" * 30 + "\n"
    for i in range(num_classes):
        count = predictions_counter[i]
        perc = (count/total) * 100
        bar = "█" * int(perc/2)  # Visual bar
        logs += f"Class {i:2d}: {count:4d} ({perc:5.1f}%) {bar}\n"
    logs += "=" * 30 + "\n"
    logs += f"Total: {total:d} samples"

    return logs
