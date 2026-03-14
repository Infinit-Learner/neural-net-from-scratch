import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

from typing import Tuple, List

class Activations():
    def __init__(self, function_choice: str):
        self.function_choice = function_choice.lower()
        self.function_options = ['relu', 'leaky_relu', 'sigmoid', 'softmax', 'tanh', 'linear']
        self.invalid_func = False 
        self.is_in_options()
        self.func_dict = {
            'relu': [self.relu_forward, self.relu_backward  ],
            'leaky_relu': [self.leaky_relu_forward, self.leaky_relu_backward],
            'sigmoid': [self.sigmoid_forward, self.sigmoid_backward],
            'softmax': [self.softmax_forward, self.softmax_backwards],
            'tanh': [self.tanh_forward, self.tanh_backward],
            'linear': [self.linear_forward, self.linear_backward]
        }
        self.selected_methods = self.func_dict[self.function_choice]
        

    def is_in_options(self):
        if self.function_choice not in self.function_options:
            self.invalid_func = True 
            raise ValueError('Invalid Activation Function')
        else:
            self.invalid_func = False 

    def forward(self, z: np.ndarray):
        if self.invalid_func == True: 
            return " Invalid Activation Function Inputted "

        else: 
            activations = self.selected_methods[0](z = z)
            return activations
    def backward(self, z: np.ndarray):
        if self.invalid_func == True: 
            return " Invalid Activation Function Inputted "

        else: 
            derivative = self.selected_methods[1](z = z)
            return derivative

    def relu_forward(self, z: np.ndarray):
        '''
        Conducts the ReLU activation function on the inputed value for the forward pass   

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: neuron activations
        '''
        relu_z = np.maximum(z, 0)
        return relu_z
    


    def relu_backward(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the derivative of ReLU activation function on the inputed value for back propagation

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: deriavtive results 
        '''
        relu_mask = z > 0  
        relu_derivative = relu_mask.astype(int)
        return relu_derivative 

    def sigmoid_forward(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the Sigmoid activation function on the inputed value for the forward pass   

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: neuron activations
        '''
        sigmoid_z = 1/(1 + np.exp(-1*z))
        return sigmoid_z
    
    def sigmoid_backward(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the derivative of ReLu activation function on the inputed value for back propagation

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: deriavtive results 
        '''
        sigmoid_z = self.sigmoid_forward(z)
        sigmoid_derivative = sigmoid_z * (1- sigmoid_z)
        return sigmoid_derivative
    
    def leaky_relu_forward(self, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        '''
        Conducts the Leaky ReLU activation function on the inputed value for the forward pass   

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 
            aplha (float): A hyperparameter that controls the slope for negative inputs 
        Return: 
            np.ndarray: neuron activations
        '''
        leaky_relu_z = np.maximum(z, alpha * z )
        return leaky_relu_z

    def leaky_relu_backward(self, z: np.ndarray, alpha: float = 0.01)-> np.ndarray:
        '''
        Conducts the derivative of ReLU activation function on the inputed value for back propagation

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: deriavtive results 
        ''' 
        leaky_relu_derivative = np.where(z > 0, 1, alpha)
        return leaky_relu_derivative 
    
    def tanh_forward(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the Tanh activation function on the inputed value for the forward pass   

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 
            aplha (float): A hyperparameter that controls the slope for negative inputs 
        Return: 
            np.ndarray: neuron activations
        '''
        tanh_z = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        return tanh_z
    
    def tanh_backward(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the derivative of ReLu activation function on the inputed value for back propagation

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: deriavtive results 
        ''' 
        tanh_z = self.tanh_forward(z)
        tanh_derivative = 1 - (tanh_z ** 2 )
        return tanh_derivative
    
    def softmax_forward(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the Softmax activation function on the inputed value for the forward pass   

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 
        Return: 
            np.ndarray: neuron activations
        '''
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)
        
    
    def softmax_backwards(self, z: np.ndarray)-> np.ndarray:
        '''
        Conducts the derivative of Softmax activation function on the inputed value for back propagation

        Args: 
            z (np.ndarray): Weighted sum computed through the weights and biases 

        Return: 
            np.ndarray: derivative results 
        ''' 
        s = self.softmax_forward(z).reshape(-1, 1)
        return np.diagflat(s) - s @ s.T 

    def linear_forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def linear_backward(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)
       