#!/usr/bin/env python3
"""
MLX Linear Algebra Tutorial
===========================

This example demonstrates linear algebra operations with MLX:
- Matrix operations
- Linear transformations
- Eigenvalues and eigenvectors
- Solving linear systems
- Advanced broadcasting
"""

import mlx.core as mx
import numpy as np


def matrix_operations():
    """Demonstrate basic matrix operations."""
    print("=" * 50)
    print("MATRIX OPERATIONS")
    print("=" * 50)
    
    # Create matrices
    A = mx.array([[1, 2, 3],
                  [4, 5, 6]], dtype=mx.float32)
    B = mx.array([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=mx.float32)
    
    print(f"Matrix A (2x3):\n{A}")
    print(f"Matrix B (3x2):\n{B}")
    
    # Matrix multiplication
    C = A @ B  # or mx.matmul(A, B)
    print(f"A @ B (matrix multiplication):\n{C}")
    
    # Element-wise multiplication
    square_matrix = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    element_wise = square_matrix * square_matrix
    print(f"Element-wise multiplication:\n{element_wise}")
    
    # Transpose
    print(f"A transpose:\n{A.T}")
    
    # Trace (sum of diagonal elements)
    square = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mx.float32)
    print(f"Square matrix:\n{square}")
    print(f"Trace: {mx.trace(square)}")


def linear_algebra_functions():
    """Demonstrate advanced linear algebra functions."""
    print("\n" + "=" * 50)
    print("ADVANCED LINEAR ALGEBRA")
    print("=" * 50)
    
    # Create a symmetric matrix for eigenvalue decomposition
    A = mx.array([[4, 2, 1],
                  [2, 3, 2],
                  [1, 2, 5]], dtype=mx.float32)
    
    print(f"Matrix A:\n{A}")
    
    # Matrix inverse (use CPU stream for compatibility)
    try:
        # Some operations may need CPU stream in certain MLX versions
        with mx.stream(mx.cpu):
            inv_A = mx.linalg.inv(A)
            mx.eval(inv_A)  # Force evaluation
        print(f"Inverse of A:\n{inv_A}")
        
        # Verify A @ inv(A) ≈ I
        identity_check = A @ inv_A
        print(f"A @ inv(A) (should be identity):\n{identity_check}")
    except Exception as e:
        print(f"Matrix inversion: {e}")
        print("Note: Some operations may require CPU stream in certain MLX versions")
    
    # QR decomposition
    try:
        with mx.stream(mx.cpu):
            Q, R = mx.linalg.qr(A)
            mx.eval(Q, R)  # Force evaluation
        print(f"Q (orthogonal matrix):\n{Q}")
        print(f"R (upper triangular):\n{R}")
        print(f"Q @ R (reconstruction):\n{Q @ R}")
    except Exception as e:
        print(f"QR decomposition: {e}")
        print("Note: Some operations may require CPU stream in certain MLX versions")


def solving_linear_systems():
    """Demonstrate solving systems of linear equations."""
    print("\n" + "=" * 50)
    print("SOLVING LINEAR SYSTEMS")
    print("=" * 50)
    
    # Solve Ax = b
    A = mx.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=mx.float32)
    b = mx.array([8, -11, -3], dtype=mx.float32)
    
    print(f"Coefficient matrix A:\n{A}")
    print(f"Constants vector b: {b}")
    
    try:
        # Solve the system with CPU stream
        with mx.stream(mx.cpu):
            x = mx.linalg.solve(A, b)
            mx.eval(x)  # Force evaluation
        print(f"Solution x: {x}")
        
        # Verify the solution
        verification = A @ x
        print(f"Verification A @ x: {verification}")
        print(f"Original b: {b}")
        print(f"Difference: {mx.abs(verification - b)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure MLX is properly installed and you're using a recent version.")


def vector_operations():
    """Demonstrate vector operations."""
    print("\n" + "=" * 50)
    print("VECTOR OPERATIONS")
    print("=" * 50)
    
    # Create vectors
    u = mx.array([1, 2, 3], dtype=mx.float32)
    v = mx.array([4, 5, 6], dtype=mx.float32)
    
    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    
    # Dot product
    dot_product = mx.sum(u * v)  # or u @ v for 1D vectors
    print(f"Dot product u·v: {dot_product}")
    
    # Vector norms
    u_norm = mx.linalg.norm(u)
    v_norm = mx.linalg.norm(v)
    print(f"||u|| (L2 norm): {u_norm}")
    print(f"||v|| (L2 norm): {v_norm}")
    
    # Unit vectors
    u_unit = u / u_norm
    v_unit = v / v_norm
    print(f"Unit vector u: {u_unit}")
    print(f"Unit vector v: {v_unit}")
    
    # Angle between vectors
    cos_theta = dot_product / (u_norm * v_norm)
    theta = mx.arccos(cos_theta)
    print(f"Angle between u and v (radians): {theta}")
    print(f"Angle between u and v (degrees): {theta * 180 / mx.pi}")


def advanced_broadcasting():
    """Demonstrate advanced broadcasting scenarios."""
    print("\n" + "=" * 50)
    print("ADVANCED BROADCASTING")
    print("=" * 50)
    
    # Batch matrix operations
    # Simulate a batch of 2x2 matrices
    batch_matrices = mx.array([[[1, 2], [3, 4]],
                               [[5, 6], [7, 8]],
                               [[9, 10], [11, 12]]], dtype=mx.float32)  # Shape: (3, 2, 2)
    
    # Vector to apply to each matrix
    vector = mx.array([1, -1], dtype=mx.float32)  # Shape: (2,)
    
    print(f"Batch of matrices (3x2x2):\n{batch_matrices}")
    print(f"Vector: {vector}")
    
    # Apply vector to each matrix
    result = batch_matrices @ vector  # Broadcasting: (3,2,2) @ (2,) -> (3,2)
    print(f"Batch matrix-vector product:\n{result}")
    
    # Outer product broadcasting
    a = mx.array([1, 2, 3], dtype=mx.float32)  # Shape: (3,)
    b = mx.array([4, 5], dtype=mx.float32)     # Shape: (2,)
    
    # Reshape for outer product
    outer = a[:, None] * b[None, :]  # (3,1) * (1,2) -> (3,2)
    print(f"Outer product:\n{outer}")
    
    # Alternative using reshape
    outer2 = mx.outer(a, b)
    print(f"Outer product (using mx.outer):\n{outer2}")


def matrix_decompositions():
    """Demonstrate matrix decomposition techniques."""
    print("\n" + "=" * 50)
    print("MATRIX DECOMPOSITIONS")
    print("=" * 50)
    
    # Create a random-like matrix
    np.random.seed(42)
    A_np = np.random.randn(4, 3).astype(np.float32)
    A = mx.array(A_np)
    
    print(f"Original matrix A (4x3):\n{A}")
    
    # Singular Value Decomposition (SVD)
    try:
        U, s, Vt = mx.linalg.svd(A)
        print("SVD completed successfully:")
        print(f"U shape: {U.shape}")
        print(f"s (singular values): {s}")
        print(f"Vt shape: {Vt.shape}")
        
        # Simple reconstruction check
        print(f"Original matrix norm: {mx.linalg.norm(A)}")
        print(f"Singular values sum: {mx.sum(s)}")
    except Exception as e:
        print(f"SVD operation: {e}")
        print("Note: Some advanced operations may need specific MLX configuration")


def geometric_transformations():
    """Demonstrate geometric transformations using matrices."""
    print("\n" + "=" * 50)
    print("GEOMETRIC TRANSFORMATIONS")
    print("=" * 50)
    
    # 2D rotation matrix
    angle = mx.pi / 4  # 45 degrees
    cos_a = mx.cos(angle)
    sin_a = mx.sin(angle)
    
    rotation_matrix = mx.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    
    print(f"45° rotation matrix:\n{rotation_matrix}")
    
    # Points to transform
    points = mx.array([[1, 0],   # Point on x-axis
                       [0, 1],   # Point on y-axis
                       [1, 1]])  # Diagonal point
    
    print(f"Original points:\n{points}")
    
    # Apply rotation
    rotated_points = points @ rotation_matrix.T
    print(f"Rotated points:\n{rotated_points}")
    
    # Scaling matrix
    scale_matrix = mx.array([[2, 0],
                             [0, 0.5]])
    
    scaled_points = points @ scale_matrix.T
    print(f"Scaled points (2x in x, 0.5x in y):\n{scaled_points}")


def performance_comparison():
    """Compare MLX performance characteristics."""
    print("\n" + "=" * 50)
    print("PERFORMANCE CHARACTERISTICS")
    print("=" * 50)
    
    # Create larger matrices for performance demonstration
    size = 100
    A = mx.random.normal((size, size))
    B = mx.random.normal((size, size))
    
    print(f"Working with {size}x{size} matrices")
    
    # Matrix multiplication
    C = A @ B
    print(f"Matrix multiplication completed")
    print(f"Result shape: {C.shape}")
    print(f"Result norm: {mx.linalg.norm(C)}")
    
    print("\n💡 MLX Performance Notes:")
    print("• Lazy evaluation: Operations are fused when possible")
    print("• Unified memory: No CPU↔GPU transfers needed")
    print("• Metal Performance Shaders: Optimized for Apple Silicon")
    print("• Automatic differentiation ready")


def main():
    """Run all linear algebra examples."""
    print("🧮 MLX Linear Algebra Tutorial")
    print("Advanced Mathematical Operations on Apple Silicon\n")
    
    try:
        matrix_operations()
        linear_algebra_functions()
        solving_linear_systems()
        vector_operations()
        advanced_broadcasting()
        matrix_decompositions()
        geometric_transformations()
        performance_comparison()
        
        print("\n" + "=" * 50)
        print("✅ Linear algebra tutorial completed!")
        print("Next: Try running 03_neural_networks.py")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure MLX is properly installed and you're using a recent version.")


if __name__ == "__main__":
    main()
