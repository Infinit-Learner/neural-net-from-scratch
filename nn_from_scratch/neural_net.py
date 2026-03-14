import numpy as np
import matplotlib.pyplot as plt 
import warnings
import pandas as pd 
import seaborn as sns 
from activations import Activations
from initialization import he, xavier_normal, xavier_uniform, random_normal, random_uniform, lecun_normal, zeros, orthagonal
from typing import Tuple, List

INIT_FUNCTIONS = {
            "he": he,
            "xavier_normal": xavier_normal,
            "xavier_uniform": xavier_uniform,
            "random_normal": random_normal,
            "random_uniform": random_uniform,
            "lecun_normal": lecun_normal,
            "orthagonal": orthagonal,
            "zeros": zeros
        }

DEFAULT_INIT = {
    "relu": "He",
    "leaky_relu": "He",
    "sigmoid": "xavier_uniform",
    "tanh": "xavier_uniform",


}

sample_layer = {"size": 10, "activation": "relu", "init": "he", "repeat": 3 }
network_config = [
    {}
]
class Neural_Net():
    def __init__(self, network_config: List, input_size: int  ):
        self.training_feature_matrix = None 
        self.initializations = INIT_FUNCTIONS
        self.target_matrix =  None
        self.input_size = input_size
        self.network_config = network_config
        self.activation_functions = self.activation.function_options
        self.weights = []   
        self.biases = [] 
        self.activations = []
    def construct_network(self):
        previous_layer_size = self.input_size
        for layer in self.network_config:
            n_in = previous_layer_size
            n_out = layer["size"]
            if "activation" in layer:
                activation = Activations(layer["activation"])
            else: 
                activation = Activations("relu")
                warnings.warn(
                    f"No activation function selected for layer. Defaulting to ReLu"
                )
            if "init" in layer:
                init = self.initializations[layer["init"]]
            else:
                
                if layer["activation"] not in ["tanh", "sigmoid"]:
                    selected_method = "xavier_uniform"
                    
                else:
                    selected_method = "he"
                
                init =self.initializations[selected_method]
                warnings.warn(
                    f"Initialization method not selected. Defaulting to method based on activation function. Selected method = {selected_method}"
                )
            w = init(n_in, n_out)
            b = np.zeros(n_out, 1)
            
            if "repeat" in layer:
                for i in range(layer["repeat"]):
                    self.weights.append(w)
                    self.biases.append(b)
                    self.activations.append(activation)
            else: 
                self.weights.append(w)
                self.biases.append(b)
                self.activations.append(activation)
            previous_layer_size = n_out

        


