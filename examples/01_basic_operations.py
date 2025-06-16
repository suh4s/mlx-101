#!/usr/bin/env python3
"""
MLX Basic Operations Tutorial
============================

This example demonstrates fundamental MLX operations including:
- Array creation
- Basic mathematical operations  
- Type handling
- Memory management
- Conversion between MLX and NumPy
"""

import mlx.core as mx
import numpy as np


def array_creation_examples():
    """Demonstrate different ways to create MLX arrays."""
    print("=" * 50)
    print("ARRAY CREATION EXAMPLES")
    print("=" * 50)
    
    # Create arrays from Python lists
    arr1 = mx.array([1, 2, 3, 4, 5])
    print(f"From list: {arr1}")
    print(f"Shape: {arr1.shape}, Dtype: {arr1.dtype}")
    
    # Create arrays with specific data types
    arr2 = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    print(f"Float32 array: {arr2}")
    
    # Create multidimensional arrays
    matrix = mx.array([[1, 2, 3], [4, 5, 6]])
    print(f"2D array:\n{matrix}")
    print(f"Shape: {matrix.shape}")
    
    # Create arrays with special values
    zeros = mx.zeros((3, 4))
    ones = mx.ones((2, 3))
    eye = mx.eye(3)
    
    print(f"Zeros (3x4):\n{zeros}")
    print(f"Ones (2x3):\n{ones}")
    print(f"Identity (3x3):\n{eye}")
    
    # Create arrays with ranges
    range_arr = mx.arange(0, 10, 2)
    linspace_arr = mx.linspace(0, 1, 5)
    
    print(f"Range (0 to 10, step 2): {range_arr}")
    print(f"Linspace (0 to 1, 5 points): {linspace_arr}")


def basic_operations():
    """Demonstrate basic mathematical operations."""
    print("\n" + "=" * 50)
    print("BASIC MATHEMATICAL OPERATIONS")
    print("=" * 50)
    
    # Create sample arrays
    a = mx.array([1, 2, 3, 4])
    b = mx.array([5, 6, 7, 8])
    
    print(f"Array a: {a}")
    print(f"Array b: {b}")
    
    # Element-wise operations
    print(f"Addition (a + b): {a + b}")
    print(f"Subtraction (a - b): {a - b}")
    print(f"Multiplication (a * b): {a * b}")
    print(f"Division (a / b): {a / b}")
    print(f"Power (a ** 2): {a ** 2}")
    
    # Scalar operations
    print(f"Scalar addition (a + 10): {a + 10}")
    print(f"Scalar multiplication (a * 3): {a * 3}")
    
    # Mathematical functions
    x = mx.array([0, 1, 2, 3, 4])
    print(f"Original: {x}")
    print(f"Square root: {mx.sqrt(x.astype(mx.float32))}")
    print(f"Exponential: {mx.exp(x.astype(mx.float32))}")
    print(f"Sine: {mx.sin(x.astype(mx.float32))}")


def array_indexing_and_slicing():
    """Demonstrate array indexing and slicing."""
    print("\n" + "=" * 50)
    print("ARRAY INDEXING AND SLICING")
    print("=" * 50)
    
    # Create a 2D array
    matrix = mx.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
    
    print(f"Original matrix:\n{matrix}")
    
    # Basic indexing
    print(f"Element at [0, 0]: {matrix[0, 0]}")
    print(f"Element at [1, 2]: {matrix[1, 2]}")
    
    # Row and column access
    print(f"First row: {matrix[0, :]}")
    print(f"Second column: {matrix[:, 1]}")
    
    # Slicing
    print(f"First 2 rows, first 3 columns:\n{matrix[:2, :3]}")
    print(f"Every other element in first row: {matrix[0, ::2]}")


def reductions_and_aggregations():
    """Demonstrate reduction operations."""
    print("\n" + "=" * 50)
    print("REDUCTIONS AND AGGREGATIONS")
    print("=" * 50)
    
    # Create sample data
    data = mx.array([[1, 2, 3], 
                     [4, 5, 6], 
                     [7, 8, 9]])
    
    print(f"Data:\n{data}")
    
    # Global reductions
    print(f"Sum of all elements: {mx.sum(data)}")
    print(f"Mean of all elements: {mx.mean(data.astype(mx.float32))}")
    print(f"Maximum element: {mx.max(data)}")
    print(f"Minimum element: {mx.min(data)}")
    
    # Axis-wise reductions
    print(f"Sum along axis 0 (columns): {mx.sum(data, axis=0)}")
    print(f"Sum along axis 1 (rows): {mx.sum(data, axis=1)}")
    print(f"Mean along axis 0: {mx.mean(data.astype(mx.float32), axis=0)}")


def mlx_numpy_interop():
    """Demonstrate MLX and NumPy interoperability."""
    print("\n" + "=" * 50)
    print("MLX ↔ NUMPY INTEROPERABILITY")
    print("=" * 50)
    
    # Start with NumPy array
    np_array = np.array([1, 2, 3, 4, 5])
    print(f"NumPy array: {np_array}")
    print(f"NumPy type: {type(np_array)}")
    
    # Convert to MLX
    mlx_array = mx.array(np_array)
    print(f"MLX array: {mlx_array}")
    print(f"MLX type: {type(mlx_array)}")
    
    # Perform MLX operations
    mlx_result = mlx_array * 2 + 1
    print(f"MLX operation result: {mlx_result}")
    
    # Convert back to NumPy
    np_result = np.array(mlx_result)
    print(f"Back to NumPy: {np_result}")
    print(f"NumPy type: {type(np_result)}")
    
    # Performance note
    print("\nNote: MLX arrays live in unified memory, accessible by both CPU and GPU!")


def broadcasting_examples():
    """Demonstrate MLX broadcasting rules."""
    print("\n" + "=" * 50)
    print("BROADCASTING EXAMPLES")
    print("=" * 50)
    
    # Different shaped arrays
    a = mx.array([[1], [2], [3]])  # Shape: (3, 1)
    b = mx.array([10, 20, 30, 40]) # Shape: (4,)
    
    print(f"Array a (3x1):\n{a}")
    print(f"Array b (4,): {b}")
    
    # Broadcasting in action
    result = a + b  # Broadcasted to (3, 4)
    print(f"a + b (broadcasted to 3x4):\n{result}")
    
    # Matrix-vector operations
    matrix = mx.array([[1, 2, 3], [4, 5, 6]])
    vector = mx.array([10, 20, 30])
    
    print(f"\nMatrix (2x3):\n{matrix}")
    print(f"Vector (3,): {vector}")
    print(f"Matrix + Vector:\n{matrix + vector}")


def memory_and_device_info():
    """Show memory and device information."""
    print("\n" + "=" * 50)
    print("MEMORY AND DEVICE INFORMATION")
    print("=" * 50)
    
    # Create an array
    arr = mx.array([1, 2, 3, 4, 5])
    
    # Device information
    print(f"Array: {arr}")
    print(f"Array size: {arr.size} elements")
    print(f"Array dtype: {arr.dtype}")
    print(f"Array shape: {arr.shape}")
    print(f"Array ndim: {arr.ndim}")
    
    # MLX uses unified memory - arrays are accessible by both CPU and GPU
    print("\nMLX Memory Model:")
    print("• Unified memory shared between CPU and GPU")
    print("• No explicit memory transfers needed")
    print("• Lazy evaluation for efficiency")


def performance_tips():
    """Share performance optimization tips."""
    print("\n" + "=" * 50)
    print("PERFORMANCE TIPS")
    print("=" * 50)
    
    print("🚀 MLX Performance Best Practices:")
    print("1. Use vectorized operations instead of loops")
    print("2. Minimize data type conversions")
    print("3. Leverage broadcasting for element-wise operations")
    print("4. Use in-place operations when possible")
    print("5. Take advantage of lazy evaluation")
    
    # Example: Vectorized vs loop (conceptual)
    print("\n✅ Good - Vectorized:")
    print("result = a * 2 + b")
    
    print("\n❌ Avoid - Element-wise loops:")
    print("result = [a[i] * 2 + b[i] for i in range(len(a))]")


def main():
    """Run all examples."""
    print("🍎 MLX Basic Operations Tutorial")
    print("Apple's Machine Learning Framework for Apple Silicon\n")
    
    try:
        array_creation_examples()
        basic_operations()
        array_indexing_and_slicing()
        reductions_and_aggregations()
        mlx_numpy_interop()
        broadcasting_examples()
        memory_and_device_info()
        performance_tips()
        
        print("\n" + "=" * 50)
        print("✅ Tutorial completed successfully!")
        print("Next: Try running 02_linear_algebra.py")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure MLX is properly installed in your environment.")


if __name__ == "__main__":
    main()
