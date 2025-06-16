#!/usr/bin/env python3
"""
MLX Neural Networks Tutorial
============================

This example demonstrates building neural networks with MLX:
- Simple perceptron
- Multi-layer neural network
- Activation functions
- Loss functions
- Basic training loop
- Gradient computation
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def activation_functions():
    """Demonstrate common activation functions."""
    print("=" * 50)
    print("ACTIVATION FUNCTIONS")
    print("=" * 50)
    
    # Create input data
    x = mx.linspace(-5, 5, 100)
    
    # Common activation functions
    relu = mx.maximum(x, 0)
    sigmoid = 1 / (1 + mx.exp(-x))
    tanh = mx.tanh(x)
    
    print(f"Input range: {x[0]:.2f} to {x[-1]:.2f}")
    print(f"ReLU max: {mx.max(relu):.2f}, min: {mx.min(relu):.2f}")
    print(f"Sigmoid max: {mx.max(sigmoid):.2f}, min: {mx.min(sigmoid):.2f}")
    print(f"Tanh max: {mx.max(tanh):.2f}, min: {mx.min(tanh):.2f}")
    
    # Demonstrate derivatives (for backpropagation understanding)
    def relu_derivative(x):
        return (x > 0).astype(mx.float32)
    
    def sigmoid_derivative(x):
        s = 1 / (1 + mx.exp(-x))
        return s * (1 - s)
    
    print("\nActivation function properties:")
    print("• ReLU: Fast, but can cause dying neurons")
    print("• Sigmoid: Smooth, but prone to vanishing gradients")
    print("• Tanh: Zero-centered, good for hidden layers")


class SimplePerceptron:
    """A simple single-layer perceptron."""
    
    def __init__(self, input_size, output_size):
        """Initialize perceptron with random weights."""
        self.weights = mx.random.normal((input_size, output_size)) * 0.1
        self.bias = mx.zeros((output_size,))
    
    def forward(self, x):
        """Forward pass through perceptron."""
        return x @ self.weights + self.bias
    
    def predict(self, x):
        """Make predictions (with sigmoid activation)."""
        logits = self.forward(x)
        return 1 / (1 + mx.exp(-logits))


class MLPNetwork:
    """A multi-layer perceptron (MLP) neural network."""
    
    def __init__(self, layer_sizes):
        """Initialize MLP with given layer sizes."""
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Xavier/Glorot initialization
            scale = mx.sqrt(2.0 / (input_size + output_size))
            weights = mx.random.normal((input_size, output_size)) * scale
            bias = mx.zeros((output_size,))
            
            self.layers.append({
                'weights': weights,
                'bias': bias
            })
    
    def forward(self, x):
        """Forward pass through the network."""
        activations = [x]
        current = x
        
        for i, layer in enumerate(self.layers):
            # Linear transformation
            linear = current @ layer['weights'] + layer['bias']
            
            # Apply activation (ReLU for hidden layers, none for output)
            if i < len(self.layers) - 1:
                current = mx.maximum(linear, 0)  # ReLU
            else:
                current = linear  # No activation for output layer
            
            activations.append(current)
        
        return current, activations
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            params.extend([layer['weights'], layer['bias']])
        return params


def loss_functions():
    """Demonstrate common loss functions."""
    print("\n" + "=" * 50)
    print("LOSS FUNCTIONS")
    print("=" * 50)
    
    # Sample predictions and targets
    predictions = mx.array([0.8, 0.3, 0.9, 0.1])
    targets = mx.array([1.0, 0.0, 1.0, 0.0])
    
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    
    # Mean Squared Error (MSE)
    mse = mx.mean((predictions - targets) ** 2)
    print(f"MSE Loss: {mse}")
    
    # Binary Cross-Entropy
    epsilon = 1e-7  # For numerical stability
    predictions_clipped = mx.clip(predictions, epsilon, 1 - epsilon)
    bce = -mx.mean(targets * mx.log(predictions_clipped) + 
                   (1 - targets) * mx.log(1 - predictions_clipped))
    print(f"Binary Cross-Entropy Loss: {bce}")
    
    # Mean Absolute Error (MAE)
    mae = mx.mean(mx.abs(predictions - targets))
    print(f"MAE Loss: {mae}")


def gradient_computation_example():
    """Demonstrate gradient computation."""
    print("\n" + "=" * 50)
    print("GRADIENT COMPUTATION")
    print("=" * 50)
    
    # Simple function: f(x) = x^2 + 2x + 1
    def simple_function(x):
        return x**2 + 2*x + 1
    
    # Analytical derivative: f'(x) = 2x + 2
    def analytical_derivative(x):
        return 2*x + 2
    
    # Test point
    x = mx.array(3.0)
    
    # Numerical gradient using finite differences
    h = 1e-5
    numerical_grad = (simple_function(x + h) - simple_function(x - h)) / (2 * h)
    
    # Analytical gradient
    analytical_grad = analytical_derivative(x)
    
    print(f"Function value at x=3: {simple_function(x)}")
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Difference: {mx.abs(numerical_grad - analytical_grad)}")


def simple_training_example():
    """Demonstrate a simple training loop."""
    print("\n" + "=" * 50)
    print("SIMPLE TRAINING EXAMPLE")
    print("=" * 50)
    
    # Generate synthetic linear data: y = 2x + 1 + noise
    np.random.seed(42)
    n_samples = 100
    x_data = np.random.randn(n_samples, 1).astype(np.float32)
    y_data = (2 * x_data + 1 + 0.1 * np.random.randn(n_samples, 1)).astype(np.float32)
    
    # Convert to MLX arrays
    X = mx.array(x_data)
    y = mx.array(y_data)
    
    print(f"Training data: {n_samples} samples")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Create a simple linear model
    model = SimplePerceptron(1, 1)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 100
    
    print(f"Initial weights: {model.weights[0, 0]:.4f}")
    print(f"Initial bias: {model.bias[0]:.4f}")
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        
        # Compute loss (MSE)
        loss = mx.mean((predictions - y) ** 2)
        losses.append(float(loss))
        
        # Compute gradients manually (for educational purposes)
        error = predictions - y
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * mx.sum(error)
        
        # Update parameters
        model.weights = model.weights - learning_rate * dw
        model.bias = model.bias - learning_rate * db
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    print(f"Final weights: {model.weights[0, 0]:.4f} (should be ≈ 2.0)")
    print(f"Final bias: {model.bias[0]:.4f} (should be ≈ 1.0)")
    print(f"Final loss: {losses[-1]:.6f}")


def mlp_classification_example():
    """Demonstrate MLP for binary classification."""
    print("\n" + "=" * 50)
    print("MLP BINARY CLASSIFICATION")
    print("=" * 50)
    
    # Generate synthetic classification data (XOR-like problem)
    np.random.seed(42)
    n_samples = 200
    
    # Create XOR-like data
    x1 = np.random.randn(n_samples//2, 2) + [1, 1]
    x2 = np.random.randn(n_samples//2, 2) + [-1, -1]
    X_pos = np.vstack([x1[:n_samples//4], x2[:n_samples//4]])
    
    x3 = np.random.randn(n_samples//2, 2) + [1, -1]
    x4 = np.random.randn(n_samples//2, 2) + [-1, 1]
    X_neg = np.vstack([x3[:n_samples//4], x4[:n_samples//4]])
    
    X_data = np.vstack([X_pos, X_neg]).astype(np.float32)
    y_data = np.vstack([np.ones((n_samples//2, 1)), 
                       np.zeros((n_samples//2, 1))]).astype(np.float32)
    
    # Convert to MLX
    X = mx.array(X_data)
    y = mx.array(y_data)
    
    print(f"Classification data: {n_samples} samples, 2 features")
    print(f"Positive class samples: {np.sum(y_data)}")
    print(f"Negative class samples: {n_samples - np.sum(y_data)}")
    
    # Create MLP: 2 inputs -> 10 hidden -> 1 output
    mlp = MLPNetwork([2, 10, 1])
    
    print(f"Network architecture: 2 -> 10 -> 1")
    print(f"Total parameters: {sum(p.size for p in mlp.get_parameters())}")
    
    # Simple prediction test
    predictions, _ = mlp.forward(X[:5])
    print(f"Sample predictions (before training): {predictions.flatten()}")


def neural_network_tips():
    """Share neural network best practices."""
    print("\n" + "=" * 50)
    print("NEURAL NETWORK BEST PRACTICES")
    print("=" * 50)
    
    print("🧠 MLX Neural Network Tips:")
    print("\n1. Weight Initialization:")
    print("   • Xavier/Glorot: sqrt(2/(fan_in + fan_out))")
    print("   • He: sqrt(2/fan_in) for ReLU networks")
    print("   • Avoid all zeros or all same values")
    
    print("\n2. Activation Functions:")
    print("   • ReLU: Good default for hidden layers")
    print("   • Leaky ReLU: Prevents dying neurons")
    print("   • Sigmoid/Tanh: Avoid in deep networks (vanishing gradients)")
    
    print("\n3. Learning Rate:")
    print("   • Start with 0.001-0.01")
    print("   • Use learning rate scheduling")
    print("   • Monitor loss curves")
    
    print("\n4. Regularization:")
    print("   • Dropout during training")
    print("   • L1/L2 weight decay")
    print("   • Batch normalization")
    
    print("\n5. MLX Specific:")
    print("   • Leverage unified memory architecture")
    print("   • Use vectorized operations")
    print("   • Take advantage of lazy evaluation")


def main():
    """Run all neural network examples."""
    print("🧠 MLX Neural Networks Tutorial")
    print("Building Neural Networks on Apple Silicon\n")
    
    try:
        activation_functions()
        loss_functions()
        gradient_computation_example()
        simple_training_example()
        mlp_classification_example()
        neural_network_tips()
        
        print("\n" + "=" * 50)
        print("✅ Neural networks tutorial completed!")
        print("Next: Try running 04_image_processing.py")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure all dependencies are installed.")
        print("You may need: uv pip install matplotlib")


if __name__ == "__main__":
    main()
