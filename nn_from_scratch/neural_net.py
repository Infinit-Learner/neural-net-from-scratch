import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from activations import Activations
from typing import Tuple, List


class Neural_Net():
    def __init__(self, learning_rate: float, network_config: List, activation_func: str  ):
        self.training_feature_matrix = None 
        self.target_matrix =  None 
        self.hidden_layer = hidden_layers
        self.activation = Activations()
        self.activation_functions = self.activation.function_options
        self.hidden_weights_biases = {}

