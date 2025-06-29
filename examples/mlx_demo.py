#!/usr/bin/env python3
"""
MLX Quick Demo
==============

A quick demonstration of MLX capabilities that work out of the box.
"""

import mlx.core as mx
import numpy as np


def main():
    print("🍎 MLX Quick Demo - Apple Silicon Machine Learning")
    print("=" * 60)
    
    print("\n1. 🎯 Basic Array Operations")
    print("-" * 30)
    a = mx.array([1, 2, 3, 4, 5])
    b = mx.array([2, 3, 4, 5, 6])
    result = a * b + 10
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a * b + 10 = {result}")
    
    print("\n2. 🧮 Matrix Multiplication")
    print("-" * 30)
    A = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    B = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
    C = A @ B
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A @ B =\n{C}")
    
    print("\n3. 🔬 Scientific Computing")
    print("-" * 30)
    x = mx.linspace(0, 2*mx.pi, 10)
    y = mx.sin(x)
    print(f"x = linspace(0, 2π, 10)")
    print(f"sin(x) = {y}")
    
    print("\n4. 📊 Statistics")
    print("-" * 30)
    data = mx.random.normal((1000,))
    print(f"Random normal data (1000 samples):")
    print(f"Mean: {mx.mean(data):.3f}")
    print(f"Std:  {mx.std(data):.3f}")
    print(f"Min:  {mx.min(data):.3f}")
    print(f"Max:  {mx.max(data):.3f}")
    
    print("\n5. 🔄 NumPy Interoperability")
    print("-" * 30)
    np_array = np.array([1, 4, 9, 16, 25])
    mlx_array = mx.array(np_array)
    sqrt_mlx = mx.sqrt(mlx_array.astype(mx.float32))
    back_to_numpy = np.array(sqrt_mlx)
    
    print(f"NumPy:     {np_array}")
    print(f"MLX:       {mlx_array}")
    print(f"sqrt(MLX): {sqrt_mlx}")
    print(f"Back to NumPy: {back_to_numpy}")
    
    print("\n6. 🧠 Simple Neural Network Layer")
    print("-" * 30)
    # Input data (batch_size=3, features=4)
    X = mx.random.normal((3, 4))
    
    # Weight matrix (features=4, outputs=2)
    W = mx.random.normal((4, 2)) * 0.1
    b = mx.zeros((2,))
    
    # Forward pass: linear layer
    linear_out = X @ W + b
    
    # ReLU activation
    relu_out = mx.maximum(linear_out, 0)
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Output shape: {relu_out.shape}")
    print(f"Sample output:\n{relu_out}")
    
    print("\n7. 🎨 Image-like Operations")
    print("-" * 30)
    # Create a synthetic "image"
    height, width = 8, 8
    image = mx.random.uniform(0, 1, (height, width))
    
    # Simple transformations
    flipped = image[:, ::-1]  # Horizontal flip
    cropped = image[2:6, 2:6]  # Center crop
    
    print(f"Original image shape: {image.shape}")
    print(f"Flipped shape: {flipped.shape}")
    print(f"Cropped shape: {cropped.shape}")
    print(f"Image mean: {mx.mean(image):.3f}")
    
    print("\n8. 🚀 Performance Features")
    print("-" * 30)
    print("✅ Unified Memory: CPU and GPU share memory")
    print("✅ Lazy Evaluation: Operations are fused for efficiency")
    print("✅ Apple Silicon Optimized: Uses Metal Performance Shaders")
    print("✅ Automatic Differentiation Ready")
    print("✅ NumPy Compatible API")
    
    print("\n" + "=" * 60)
    print("🎉 MLX Demo Complete!")
    print("Ready to build ML applications on Apple Silicon!")
    print("=" * 60)


if __name__ == "__main__":
    main()
