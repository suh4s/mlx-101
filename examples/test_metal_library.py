#!/usr/bin/env python3
"""
🔬 MLX Metal Library Test
Tests Metal library loading and basic GPU operations with MLX
"""

import mlx.core as mx
import mlx.nn as nn
import platform

def test_device_info():
    """Test basic device information"""
    print("🔍 Device Information")
    print("=" * 50)
    
    # Check platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    # Check MLX device
    try:
        device = mx.default_device()
        print(f"MLX Default Device: {device}")
        
        # Check if we're on Apple Silicon
        if platform.machine() == "arm64":
            print("✅ Running on Apple Silicon (ARM64)")
        else:
            print("⚠️  Not on Apple Silicon - Metal may not be available")
            
    except Exception as e:
        print(f"❌ Error getting device info: {e}")
        return False
    
    return True

def test_metal_availability():
    """Test if Metal is available and working"""
    print("\n🔧 Metal Availability Test")
    print("=" * 50)
    
    try:
        # Try to create a simple array on GPU
        x = mx.array([1.0, 2.0, 3.0])
        print(f"✅ Created MLX array: {x}")
        print(f"Array device: {x.device if hasattr(x, 'device') else 'default'}")
        
        # Force evaluation to trigger Metal usage
        mx.eval(x)
        print("✅ Array evaluation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Metal availability test failed: {e}")
        return False

def test_basic_operations():
    """Test basic MLX operations that use Metal"""
    print("\n🧮 Basic Operations Test")
    print("=" * 50)
    
    try:
        # Create test arrays
        a = mx.array([1.0, 2.0, 3.0, 4.0])
        b = mx.array([5.0, 6.0, 7.0, 8.0])
        
        print(f"Array a: {a}")
        print(f"Array b: {b}")
        
        # Test addition
        c = a + b
        mx.eval(c)
        print(f"a + b = {c}")
        
        # Test multiplication
        d = a * b
        mx.eval(d)
        print(f"a * b = {d}")
        
        # Test matrix operations
        matrix = mx.random.normal((3, 3))
        mx.eval(matrix)
        print(f"Random matrix shape: {matrix.shape}")
        
        # Test reduction
        sum_result = mx.sum(matrix)
        mx.eval(sum_result)
        print(f"Matrix sum: {sum_result}")
        
        print("✅ All basic operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        return False

def test_neural_network():
    """Test MLX neural network operations"""
    print("\n🧠 Neural Network Test")
    print("=" * 50)
    
    try:
        # Create a simple linear layer
        layer = nn.Linear(4, 2)
        
        # Create input
        x = mx.random.normal((2, 4))  # batch_size=2, input_features=4
        
        # Forward pass
        output = layer(x)
        mx.eval(output)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Layer weights shape: {layer.weight.shape}")
        print("✅ Neural network operations successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False

def test_memory_usage():
    """Test memory operations and cleanup"""
    print("\n💾 Memory Usage Test")
    print("=" * 50)
    
    try:
        # Create larger arrays to test memory
        large_array = mx.random.normal((1000, 1000))
        mx.eval(large_array)
        
        print(f"Created large array shape: {large_array.shape}")
        
        # Test memory intensive operation
        result = mx.matmul(large_array, large_array.T)
        mx.eval(result)
        
        print(f"Matrix multiplication result shape: {result.shape}")
        print("✅ Memory operations successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_gradient_operations():
    """Test gradient computations that heavily use Metal"""
    print("\n📈 Gradient Operations Test")
    print("=" * 50)
    
    try:
        # Simple function for gradient test
        def simple_function(x):
            return mx.sum(x ** 2)
        
        # Test input
        x = mx.array([1.0, 2.0, 3.0])
        
        # Compute gradient
        grad_fn = mx.grad(simple_function)
        gradient = grad_fn(x)
        mx.eval(gradient)
        
        print(f"Input: {x}")
        print(f"Gradient: {gradient}")
        print("✅ Gradient operations successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        return False

def diagnose_metal_issues():
    """Provide diagnostic information for Metal issues"""
    print("\n🔧 Metal Diagnostic Information")
    print("=" * 50)
    
    print("Common Metal library issues and solutions:")
    print()
    
    print("1. Missing Command Line Tools:")
    print("   Solution: xcode-select --install")
    print()
    
    print("2. Outdated macOS version:")
    print("   MLX requires macOS 13.3+ for optimal Metal support")
    print()
    
    print("3. Intel Mac limitations:")
    print("   MLX is optimized for Apple Silicon, limited Intel support")
    print()
    
    print("4. Metal library path issues:")
    print("   Check if /System/Library/Frameworks/Metal.framework exists")
    print()
    
    print("5. Environment issues:")
    print("   Try running in a fresh terminal or restarting")
    print()
    
    # Check system information
    try:
        import subprocess
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, text=True, timeout=10)
        if "Apple" in result.stdout:
            print("✅ System appears to be Apple hardware")
        else:
            print("⚠️  System may not be Apple hardware")
    except:
        print("ℹ️  Could not determine hardware type")

def main():
    """Run all Metal library tests"""
    print("🔬 MLX Metal Library Comprehensive Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Device Info", test_device_info),
        ("Metal Availability", test_metal_availability),
        ("Basic Operations", test_basic_operations),
        ("Neural Network", test_neural_network),
        ("Memory Usage", test_memory_usage),
        ("Gradient Operations", test_gradient_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Metal library is working correctly.")
    elif passed > 0:
        print("⚠️  Some tests passed, partial Metal functionality available.")
    else:
        print("❌ All tests failed. Metal library has issues.")
        diagnose_metal_issues()
    
    print("\n" + "=" * 60)
    print("Test completed. Check output above for detailed results.")

if __name__ == "__main__":
    main()
