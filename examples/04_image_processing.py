#!/usr/bin/env python3
"""
MLX Image Processing Tutorial
=============================

This example demonstrates image processing with MLX:
- Loading and manipulating images
- Image transformations
- Convolution operations
- Simple CNN building blocks
- Image filtering
"""

import mlx.core as mx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_sample_image():
    """Create a simple synthetic image for testing."""
    print("=" * 50)
    print("CREATING SAMPLE IMAGE")
    print("=" * 50)
    
    # Create a simple pattern
    height, width = 64, 64
    x = mx.linspace(-2, 2, width)
    y = mx.linspace(-2, 2, height)
    
    # Create meshgrid
    X, Y = mx.meshgrid(x, y)
    
    # Create a pattern (Gaussian-like)
    pattern = mx.exp(-(X**2 + Y**2))
    
    # Add some noise
    noise = mx.random.normal(pattern.shape) * 0.1
    image = pattern + noise
    
    # Normalize to [0, 1]
    image = (image - mx.min(image)) / (mx.max(image) - mx.min(image))
    
    print(f"Created synthetic image: {image.shape}")
    print(f"Image range: [{mx.min(image):.3f}, {mx.max(image):.3f}]")
    
    return image


def basic_image_operations():
    """Demonstrate basic image operations."""
    print("\n" + "=" * 50)
    print("BASIC IMAGE OPERATIONS")
    print("=" * 50)
    
    # Create a sample image
    image = create_sample_image()
    
    # Image statistics
    print(f"Image mean: {mx.mean(image):.3f}")
    print(f"Image std: {mx.std(image):.3f}")
    print(f"Image shape: {image.shape}")
    
    # Basic transformations
    # Brightness adjustment
    brightened = mx.clip(image + 0.2, 0, 1)
    darkened = mx.clip(image - 0.2, 0, 1)
    
    # Contrast adjustment
    contrast_enhanced = mx.clip((image - 0.5) * 1.5 + 0.5, 0, 1)
    
    # Gamma correction
    gamma_corrected = mx.power(image, 0.5)
    
    print(f"Brightened image mean: {mx.mean(brightened):.3f}")
    print(f"Darkened image mean: {mx.mean(darkened):.3f}")
    print(f"Contrast enhanced std: {mx.std(contrast_enhanced):.3f}")
    print(f"Gamma corrected mean: {mx.mean(gamma_corrected):.3f}")
    
    return image


def convolution_operations():
    """Demonstrate convolution operations."""
    print("\n" + "=" * 50)
    print("CONVOLUTION OPERATIONS")
    print("=" * 50)
    
    image = create_sample_image()
    
    # Define common kernels
    # Edge detection (Sobel)
    sobel_x = mx.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=mx.float32)
    
    sobel_y = mx.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=mx.float32)
    
    # Gaussian blur
    gaussian = mx.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=mx.float32) / 16.0
    
    # Sharpening kernel
    sharpen = mx.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]], dtype=mx.float32)
    
    print("Defined kernels:")
    print(f"Sobel X:\n{sobel_x}")
    print(f"Gaussian blur:\n{gaussian}")
    
    # Apply convolutions (simplified 2D convolution)
    def conv2d_simple(image, kernel):
        """Simple 2D convolution implementation using vectorized operations."""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Pad the image
        padded = mx.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # Initialize output as numpy array, then convert to MLX
        output_np = np.zeros((h, w))
        
        # Perform convolution
        for i in range(h):
            for j in range(w):
                patch = padded[i:i+kh, j:j+kw]
                output_np[i, j] = float(mx.sum(patch * kernel))
        
        return mx.array(output_np)
    
    # Apply different filters
    edges_x = conv2d_simple(image, sobel_x)
    edges_y = conv2d_simple(image, sobel_y)
    blurred = conv2d_simple(image, gaussian)
    sharpened = conv2d_simple(image, sharpen)
    
    # Edge magnitude
    edge_magnitude = mx.sqrt(edges_x**2 + edges_y**2)
    
    print(f"Edge magnitude range: [{mx.min(edge_magnitude):.3f}, {mx.max(edge_magnitude):.3f}]")
    print(f"Blurred image std: {mx.std(blurred):.3f}")
    print(f"Sharpened image range: [{mx.min(sharpened):.3f}, {mx.max(sharpened):.3f}]")
    
    return image, edge_magnitude, blurred


class SimpleConv2D:
    """A simple 2D convolution layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights using Xavier initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        scale = mx.sqrt(2.0 / (fan_in + fan_out))
        
        self.weights = mx.random.normal((out_channels, in_channels, kernel_size, kernel_size)) * scale
        self.bias = mx.zeros((out_channels,))
    
    def forward(self, x):
        """Forward pass through convolution layer."""
        # This is a simplified implementation
        # In practice, you'd use optimized convolution functions
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = mx.zeros((batch_size, self.out_channels, out_height, out_width))
        
        print(f"Conv2D: {x.shape} -> {output.shape}")
        return output


def cnn_building_blocks():
    """Demonstrate CNN building blocks."""
    print("\n" + "=" * 50)
    print("CNN BUILDING BLOCKS")
    print("=" * 50)
    
    # Simulate a batch of images
    batch_size, channels, height, width = 4, 3, 32, 32
    x = mx.random.normal((batch_size, channels, height, width))
    
    print(f"Input batch shape: {x.shape}")
    
    # Convolution layer
    conv1 = SimpleConv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # Max pooling (simplified)
    def max_pool2d(x, kernel_size=2, stride=2):
        """Simple max pooling implementation using numpy for assignment."""
        b, c, h, w = x.shape
        out_h = h // stride
        out_w = w // stride
        
        # Use numpy for the pooling operation
        x_np = np.array(x)
        output_np = np.zeros((b, c, out_h, out_w), dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                start_h, start_w = i * stride, j * stride
                end_h, end_w = start_h + kernel_size, start_w + kernel_size
                pool_region = x_np[:, :, start_h:end_h, start_w:end_w]
                output_np[:, :, i, j] = np.max(pool_region, axis=(2, 3))
        
        return mx.array(output_np)
    
    # ReLU activation
    def relu(x):
        return mx.maximum(x, 0)
    
    print("CNN architecture demonstration:")
    print(f"1. Input: {x.shape}")
    
    # Forward pass simulation
    conv_out = conv1.forward(x)
    print(f"2. After Conv2D: {conv_out.shape}")
    
    relu_out = relu(conv_out)
    print(f"3. After ReLU: {relu_out.shape}")
    
    pooled_out = max_pool2d(relu_out)
    print(f"4. After MaxPool: {pooled_out.shape}")


def image_transformations():
    """Demonstrate geometric image transformations."""
    print("\n" + "=" * 50)
    print("IMAGE TRANSFORMATIONS")
    print("=" * 50)
    
    image = create_sample_image()
    h, w = image.shape
    
    # Horizontal flip
    flipped_h = image[:, ::-1]
    print(f"Horizontal flip: {image.shape} -> {flipped_h.shape}")
    
    # Vertical flip
    flipped_v = image[::-1, :]
    print(f"Vertical flip: {image.shape} -> {flipped_v.shape}")
    
    # Rotation (90 degrees) - using numpy for rotation
    rotated_90 = mx.array(np.rot90(np.array(image)))
    print(f"90° rotation: {image.shape} -> {rotated_90.shape}")
    
    # Center crop
    crop_size = 32
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped = image[start_h:start_h+crop_size, start_w:start_w+crop_size]
    print(f"Center crop: {image.shape} -> {cropped.shape}")
    
    # Resize (simple nearest neighbor)
    def resize_nearest(img, new_size):
        old_h, old_w = img.shape
        new_h, new_w = new_size
        
        # Calculate scaling factors
        scale_h = old_h / new_h
        scale_w = old_w / new_w
        
        # Use numpy for resize operation
        img_np = np.array(img)
        new_img_np = np.zeros((new_h, new_w), dtype=np.float32)
        
        for i in range(new_h):
            for j in range(new_w):
                old_i = int(i * scale_h)
                old_j = int(j * scale_w)
                old_i = min(old_i, old_h - 1)
                old_j = min(old_j, old_w - 1)
                new_img_np[i, j] = img_np[old_i, old_j]
        
        return mx.array(new_img_np)
    
    resized = resize_nearest(image, (32, 32))
    print(f"Resize: {image.shape} -> {resized.shape}")


def color_space_operations():
    """Demonstrate color space operations."""
    print("\n" + "=" * 50)
    print("COLOR SPACE OPERATIONS")
    print("=" * 50)
    
    # Create a synthetic RGB image
    height, width = 32, 32
    
    # Create RGB channels
    r_channel = mx.random.uniform(0, 1, (height, width))
    g_channel = mx.random.uniform(0, 1, (height, width))
    b_channel = mx.random.uniform(0, 1, (height, width))
    
    rgb_image = mx.stack([r_channel, g_channel, b_channel], axis=-1)
    print(f"RGB image shape: {rgb_image.shape}")
    
    # Convert to grayscale (luminance)
    # Standard weights: R=0.299, G=0.587, B=0.114
    grayscale = (0.299 * rgb_image[:, :, 0] + 
                 0.587 * rgb_image[:, :, 1] + 
                 0.114 * rgb_image[:, :, 2])
    
    print(f"Grayscale shape: {grayscale.shape}")
    print(f"RGB mean: {mx.mean(rgb_image):.3f}")
    print(f"Grayscale mean: {mx.mean(grayscale):.3f}")
    
    # Channel statistics
    r_mean = mx.mean(rgb_image[:, :, 0])
    g_mean = mx.mean(rgb_image[:, :, 1])
    b_mean = mx.mean(rgb_image[:, :, 2])
    
    print(f"Channel means - R: {r_mean:.3f}, G: {g_mean:.3f}, B: {b_mean:.3f}")


def image_processing_tips():
    """Share image processing best practices."""
    print("\n" + "=" * 50)
    print("IMAGE PROCESSING BEST PRACTICES")
    print("=" * 50)
    
    print("📸 MLX Image Processing Tips:")
    print("\n1. Data Preprocessing:")
    print("   • Normalize pixel values to [0,1] or [-1,1]")
    print("   • Consider data augmentation for training")
    print("   • Handle different image sizes consistently")
    
    print("\n2. Memory Management:")
    print("   • Use MLX's unified memory efficiently")
    print("   • Process images in batches when possible")
    print("   • Consider lazy evaluation for large datasets")
    
    print("\n3. Convolution Tips:")
    print("   • Use appropriate padding to maintain spatial dimensions")
    print("   • Consider separable convolutions for efficiency")
    print("   • Use stride > 1 for downsampling")
    
    print("\n4. Performance:")
    print("   • Leverage Apple Silicon GPU for convolutions")
    print("   • Use vectorized operations over loops")
    print("   • Pre-allocate arrays when possible")
    
    print("\n5. Common Patterns:")
    print("   • Conv -> BatchNorm -> ReLU -> Pool")
    print("   • Residual connections for deep networks")
    print("   • Progressive resizing for training")


def main():
    """Run all image processing examples."""
    print("📸 MLX Image Processing Tutorial")
    print("Computer Vision with MLX on Apple Silicon\n")
    
    try:
        image = basic_image_operations()
        convolution_operations()
        cnn_building_blocks()
        image_transformations()
        color_space_operations()
        image_processing_tips()
        
        print("\n" + "=" * 50)
        print("✅ Image processing tutorial completed!")
        print("You're now ready to build computer vision applications with MLX!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure all dependencies are installed.")
        print("You may need: uv pip install pillow matplotlib")


if __name__ == "__main__":
    main()
