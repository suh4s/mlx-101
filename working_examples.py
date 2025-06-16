#!/usr/bin/env python3
"""
MLX Working Examples Demo
========================

This file contains simplified, working versions of MLX operations
that avoid GPU-only limitations in the current version.
"""

import mlx.core as mx
import numpy as np


def working_linear_algebra():
    """Demonstrate working linear algebra operations."""
    print("🧮 Working Linear Algebra Examples")
    print("=" * 50)
    
    # Basic matrix operations (these work well)
    A = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float32)
    B = mx.array([[7, 8], [9, 10], [11, 12]], dtype=mx.float32)
    
    print(f"Matrix A (2x3):\n{A}")
    print(f"Matrix B (3x2):\n{B}")
    
    # Matrix multiplication (works great on GPU)
    C = A @ B
    print(f"A @ B (matrix multiplication):\n{C}")
    
    # Element-wise operations
    square = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    squared = square * square
    print(f"Element-wise multiplication:\n{squared}")
    
    # Transpose
    print(f"A transpose:\n{A.T}")
    
    # Vector operations
    u = mx.array([1, 2, 3], dtype=mx.float32)
    v = mx.array([4, 5, 6], dtype=mx.float32)
    
    dot_product = mx.sum(u * v)
    norm_u = mx.linalg.norm(u)
    norm_v = mx.linalg.norm(v)
    
    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    print(f"Dot product: {dot_product}")
    print(f"||u||: {norm_u}")
    print(f"||v||: {norm_v}")
    
    # Angle between vectors
    cos_theta = dot_product / (norm_u * norm_v)
    angle = mx.arccos(cos_theta) * 180 / mx.pi
    print(f"Angle between vectors: {angle}°")


def working_image_processing():
    """Demonstrate working image processing operations."""
    print("\n📸 Working Image Processing Examples")
    print("=" * 50)
    
    # Create synthetic image
    height, width = 32, 32
    x = mx.linspace(-2, 2, width)
    y = mx.linspace(-2, 2, height)
    
    # Create meshgrid manually
    X = mx.broadcast_to(x[None, :], (height, width))
    Y = mx.broadcast_to(y[:, None], (height, width))
    
    # Create Gaussian pattern
    image = mx.exp(-(X**2 + Y**2))
    image = (image - mx.min(image)) / (mx.max(image) - mx.min(image))
    
    print(f"Created image: {image.shape}")
    print(f"Image range: [{mx.min(image):.3f}, {mx.max(image):.3f}]")
    
    # Basic transformations
    flipped = image[:, ::-1]
    cropped = image[8:24, 8:24]
    brightened = mx.clip(image + 0.2, 0, 1)
    
    print(f"Flipped: {flipped.shape}")
    print(f"Cropped: {cropped.shape}")
    print(f"Brightened mean: {mx.mean(brightened):.3f}")
    
    # Simple convolution kernel
    blur_kernel = mx.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=mx.float32) / 16
    
    # Manual convolution on a small patch
    patch = image[10:13, 10:13]
    convolved_value = mx.sum(patch * blur_kernel)
    
    print(f"Blur kernel:\n{blur_kernel}")
    print(f"Sample patch convolution result: {convolved_value}")
    
    # Color space operations
    rgb_image = mx.stack([image, image * 0.8, image * 0.6], axis=-1)
    grayscale = 0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
    
    print(f"RGB image: {rgb_image.shape}")
    print(f"Grayscale: {grayscale.shape}")


def working_neural_networks():
    """Demonstrate working neural network components."""
    print("\n🧠 Working Neural Network Examples")
    print("=" * 50)
    
    # Simple linear layer
    batch_size, input_size, hidden_size = 4, 3, 5
    
    # Input and parameters
    X = mx.random.normal((batch_size, input_size))
    W = mx.random.normal((input_size, hidden_size)) * 0.1
    b = mx.zeros((hidden_size,))
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Bias shape: {b.shape}")
    
    # Forward pass
    linear_out = X @ W + b
    relu_out = mx.maximum(linear_out, 0)
    
    print(f"Linear output: {linear_out.shape}")
    print(f"After ReLU: {relu_out.shape}")
    print(f"Sample ReLU output:\n{relu_out[:2]}")
    
    # Activation functions
    x = mx.linspace(-3, 3, 10)
    sigmoid = 1 / (1 + mx.exp(-x))
    tanh_out = mx.tanh(x)
    relu_out = mx.maximum(x, 0)
    
    print(f"Input range: [{x[0]:.1f}, {x[-1]:.1f}]")
    print(f"Sigmoid range: [{mx.min(sigmoid):.3f}, {mx.max(sigmoid):.3f}]")
    print(f"Tanh range: [{mx.min(tanh_out):.3f}, {mx.max(tanh_out):.3f}]")
    print(f"ReLU range: [{mx.min(relu_out):.3f}, {mx.max(relu_out):.3f}]")
    
    # Loss functions
    predictions = mx.array([0.8, 0.3, 0.9, 0.1])
    targets = mx.array([1.0, 0.0, 1.0, 0.0])
    
    mse = mx.mean((predictions - targets) ** 2)
    mae = mx.mean(mx.abs(predictions - targets))
    
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    print(f"MSE Loss: {mse}")
    print(f"MAE Loss: {mae}")


def working_advanced_operations():
    """Demonstrate advanced MLX operations that work well."""
    print("\n🚀 Advanced Working Examples")
    print("=" * 50)
    
    # Broadcasting examples
    a = mx.array([[1], [2], [3]])  # Shape: (3, 1)
    b = mx.array([10, 20, 30])     # Shape: (3,)
    
    broadcast_result = a + b
    print(f"Broadcasting (3,1) + (3,): {broadcast_result.shape}")
    print(f"Result:\n{broadcast_result}")
    
    # Batch operations
    batch_data = mx.random.normal((10, 5))  # 10 samples, 5 features
    
    batch_mean = mx.mean(batch_data, axis=0)
    batch_std = mx.std(batch_data, axis=0)
    normalized = (batch_data - batch_mean) / batch_std
    
    print(f"Batch data: {batch_data.shape}")
    print(f"Batch mean: {batch_mean}")
    print(f"Normalized mean: {mx.mean(normalized, axis=0)}")
    print(f"Normalized std: {mx.std(normalized, axis=0)}")
    
    # Indexing and slicing
    data = mx.random.normal((5, 4, 3))
    
    first_sample = data[0]
    first_two = data[:2]
    every_other = data[::2]
    specific_slice = data[1:4, :, 1]
    
    print(f"Original data: {data.shape}")
    print(f"First sample: {first_sample.shape}")
    print(f"First two: {first_two.shape}")
    print(f"Every other: {every_other.shape}")
    print(f"Specific slice: {specific_slice.shape}")


def performance_showcase():
    """Showcase MLX performance characteristics."""
    print("\n⚡ Performance Showcase")
    print("=" * 50)
    
    # Large matrix operations
    size = 512
    A = mx.random.normal((size, size))
    B = mx.random.normal((size, size))
    
    # This runs on Apple Silicon GPU with unified memory!
    C = A @ B
    
    print(f"Matrix multiplication: {size}x{size} @ {size}x{size}")
    print(f"Result shape: {C.shape}")
    print(f"Result norm: {mx.linalg.norm(C):.2f}")
    
    # Memory efficiency demonstration
    large_array = mx.random.normal((1000, 1000))
    reduced = mx.mean(large_array, axis=0)
    further_reduced = mx.sum(reduced)
    
    print(f"Large array: {large_array.shape}")
    print(f"After mean reduction: {reduced.shape}")
    print(f"Final sum: {further_reduced}")
    
    print("\n💡 What makes this fast:")
    print("• Unified memory: GPU and CPU share the same memory")
    print("• Lazy evaluation: Operations are fused automatically")
    print("• Metal Performance Shaders: Native Apple Silicon optimization")
    print("• No memory transfers: Data stays in unified memory")


def main():
    """Run all working examples."""
    print("🍎 MLX Working Examples - Apple Silicon ML")
    print("Everything in this demo works perfectly!")
    print("=" * 60)
    
    try:
        working_linear_algebra()
        working_image_processing()
        working_neural_networks()
        working_advanced_operations()
        performance_showcase()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("🚀 MLX is working perfectly on your Apple Silicon!")
        print("Ready to build amazing ML applications!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Check your MLX installation and try again.")


if __name__ == "__main__":
    main()
