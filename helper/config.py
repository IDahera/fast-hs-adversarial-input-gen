"""
This module provides helper functions to read and extract configuration settings from a YAML file.
Functions:
    read_config(config_path: str) -> dict:
        Reads a YAML configuration file and returns its contents as a dictionary.
    get_verbose_susp_comp(config: dict) -> Any:
        Retrieves the verbose setting for suspiciousness computations from the configuration.
    get_target_class(config: dict) -> Any:
        Retrieves the target class setting for suspiciousness computations from the configuration.
    get_activation_threshold(config: dict) -> Any:
        Retrieves the activation threshold setting for suspiciousness computations from the configuration.
    get_batch_size(config: dict) -> Any:
        Retrieves the batch size setting for suspiciousness computations from the configuration.
    get_num_batches(config: dict) -> Any:
        Retrieves the number of batches setting for suspiciousness computations from the configuration.
    get_target_dir(config: dict) -> Any:
        Retrieves the target directory setting for suspiciousness computations from the configuration.
    get_verbose_exp(config: dict) -> Any:
        Retrieves the verbose setting for experiments from the configuration.
    get_epochs(config: dict) -> Any:
        Retrieves the number of epochs setting for training from the configuration.
    get_results_dir(config: dict) -> Any:
        Retrieves the results directory setting for experiments from the configuration.
    get_print_images_flag(config: dict) -> Any:
        Retrieves the flag indicating whether to print images during experiments from the configuration.
    extract_model_name(key: str) -> str:
        Extracts the model name by removing the dataset prefix (mnist/fmnist/cifar10) from the given key.
    get_relevant_model_layer_configs(config: dict) -> list:
        Retrieves relevant model layer configurations from the configuration.
"""

import yaml
from typing import Dict, List, Tuple, Any

# TODO remove ANY type hints

def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class ExperimentRun():
    def __init__(self,
                 experiment_name: str,
                 scenario_name: str,
                 model_name: str,
                 model_path: str,
                 dataset_name: str,
                 layer_config: int,
                 sus_metric: str,
                 iterations: int,
                 grad_factor: float,
                 samples: int,
                 num_neurons: int,
                 print_images: bool,
                 result_dir: str):

        # Input checks
        if sus_metric not in ["ochiai", "tarantula"]:
            raise Exception(f"Invalid Suspiciousness Metric: {sus_metric}")

        # Experiment Identifier
        self.experiment_name = experiment_name
        self.scenario_name = scenario_name
        
        # Model Information
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.layer_config = layer_config
        
        # Experiment Parameters
        self.samples = samples
        self.iterations = iterations
        self.sus_metric = sus_metric
        self.num_neurons = num_neurons
        self.grad_factor = grad_factor
        
        # Documentation Parameters
        self.print_images = print_images
        self.result_dir = result_dir

    def __str__(self):
        # Return string representation of the object
        return_str = ""
        return_str += f"Experiment({self.experiment_name})/Scenario({self.scenario_name}) on model {
            self.model_name}\n"
        return_str += f"- Layer Config: {self.layer_config}\n"
        return_str += f"- Susp. Metric: {self.sus_metric}\n"
        return_str += f"- Number of Neurons: {self.num_neurons}\n"
        return_str += f"- Number of Iterations: {self.iterations}\n"
        return_str += f"- Number of Sample: {self.samples}\n"
        return_str += f"- Gradient Factor: {self.iterations}\n"
        return return_str

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExperimentRun):
            return False

        return (self.model_name == other.model_name and
                self.dataset_name == other.dataset_name and
                self.model_path == other.model_path and
                self.sus_metric == other.sus_metric and
                self.print_images == other.print_images and
                self.result_dir == other.result_dir and
                self.iterations == other.iterations and
                self.grad_factor == other.grad_factor and
                self.samples == other.samples and
                self.num_neurons == other.num_neurons and
                self.layer_config == other.layer_config)


class TrainConfig:
    def __init__(self, config_path: str):
        config = read_config(config_path)
        try:
            self.epochs = config["epochs"]

            self.models = []
            for key in config.keys():
                if key.startswith("epochs"):
                    continue

                self.models.append((key, config[key]))
        except KeyError:
            raise Exception("Train config file is not properly formatted")

    def get_models(self) -> List[Tuple[str, str]]:
        return self.models
    
    def look_up_path(self, given_model_name):
        for model_name, model_path in self.models:
            if model_name == given_model_name:
                return model_path
        return None


class SuspiciousnessConfig():
    def __init__(self, config_path: str):
        config = read_config(config_path)
        try:
            self.verbose: bool = config["verbose"]
            self.target_class: int = config["target-class"]
            self.activation_threshold: float = config["activation-threshold"]
            self.batch_size: int = config["batch-size"]
            self.batches: int = config["batches"]
            self.samples: int = self.batch_size * self.batches
            self.classes: int = config["classes"]
            self.target_dir: str = config["target-dir"]
        except KeyError:
            raise Exception(
                "Suspiciousness config file is not properly formatted")


class ExperimentConfig():
    def __init__(self, config_path: str, train_config: TrainConfig):
        self.exp_config = read_config(config_path)
        self.exp_runs: list[ExperimentRun] = self.__init_exp_runs(train_config)

    def __init_exp_runs(self,
                        train_config: TrainConfig):
        experiment_runs = []
        try:
            for experiment_key in self.exp_config.keys():
                if not experiment_key.startswith("experiment"):
                    continue

                for scenario_key in self.exp_config[experiment_key].keys():
                    num_neurons = self.exp_config[experiment_key][scenario_key]["num-neurons"]
                    iterations = self.exp_config[experiment_key][scenario_key]["iterations"]
                    grad_factor = self.exp_config[experiment_key][scenario_key]["gradient-factor"]
                    
                    sus_metric = self.exp_config[experiment_key][scenario_key]["sus-metric"]
                    print_images = self.exp_config[experiment_key][scenario_key]["print-images"]
                    result_dir = self.exp_config[experiment_key][scenario_key]["results-dir"]
                    samples = self.exp_config[experiment_key][scenario_key]["samples"]

                    for model_name, _ in train_config.get_models():
                        model_path = train_config.look_up_path(model_name)
                        model_layer_config = self.exp_config[experiment_key][scenario_key][model_name + "-layer-config"]
                        skip_model = self.exp_config[experiment_key][scenario_key]["skip-" + model_name]

                        new_run = ExperimentRun(experiment_name=experiment_key,
                                                scenario_name=scenario_key,
                                                model_name=model_name,
                                                dataset_name=model_name.split(
                                                    "-")[0],
                                                sus_metric=sus_metric,
                                                iterations=iterations,
                                                grad_factor=grad_factor,
                                                model_path=model_path,
                                                num_neurons=num_neurons,
                                                print_images=print_images,
                                                samples=samples,
                                                result_dir=result_dir,
                                                layer_config=model_layer_config)

                        if not skip_model and new_run not in experiment_runs:
                            experiment_runs.append(new_run)

            return experiment_runs

        except KeyError:
            raise Exception("Experiment config file is not properly formatted")

    def get_relevant_model_layer_configs(self) -> List[Tuple[str, Any]]:
        relevant_model_configs = [(exp_run.model_name, exp_run.layer_config) for exp_run in self.exp_runs]
        return list(set(relevant_model_configs))


def extract_model_name(key: str) -> str:
    """Extract model name by removing dataset prefix (mnist/fmnist/cifar10)"""
    for prefix in ['mnist-', 'fmnist-', 'cifar10-']:
        if key.startswith(prefix):
            return key.replace(prefix, '')
    return None


def get_relevant_model_layer_configs(config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    relevant_models = []
    for model_name in config["train-config"]:
        if model_name == "epochs":
            continue

        relevant_models.append(model_name)

    relevant_model_configs = []
    for experiment_id in config.keys():
        if not experiment_id.startswith("experiment"):
            continue

        for scenario_id in config[experiment_id].keys():
            for model_name in relevant_models:
                relevant_model_configs.append((model_name,
                                               config[experiment_id][scenario_id][model_name + "-layer-config"]))

    return list(set(relevant_model_configs))