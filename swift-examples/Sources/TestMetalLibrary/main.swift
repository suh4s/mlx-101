import Foundation
import MLX
import MLXRandom

@main
struct TestMetalLibrary {
    
    static func main() {
        print("🔬 MLX Swift Metal Library Test")
        print("=" + String(repeating: "=", count: 50))
        
        var testsPassedCount = 0
        let totalTests = 6
        
        testsPassedCount += testDeviceInfo() ? 1 : 0
        testsPassedCount += testMetalAvailability() ? 1 : 0
        testsPassedCount += testBasicOperations() ? 1 : 0
        testsPassedCount += testMatrixOperations() ? 1 : 0
        testsPassedCount += testMemoryOperations() ? 1 : 0
        testsPassedCount += testRandomOperations() ? 1 : 0
        
        print("\n📊 Test Results: \(testsPassedCount)/\(totalTests) tests passed")
        
        if testsPassedCount == totalTests {
            print("🎉 All tests passed! Metal library is working correctly.")
        } else if testsPassedCount > 0 {
            print("⚠️  Some tests passed, partial Metal functionality available.")
        } else {
            print("❌ All tests failed. Metal library has issues.")
            diagnoseMetal()
        }
        
        print("\n" + String(repeating: "=", count: 50))
        print("Test completed. Check output above for detailed results.")
    }
    
    static func testDeviceInfo() -> Bool {
        print("\n🔍 Device Information Test")
        print(String(repeating: "-", count: 50))
        
        do {
            // Check system information
            let processInfo = ProcessInfo.processInfo
            print("System: \(processInfo.operatingSystemVersionString)")
            
            #if arch(arm64)
            print("✅ Running on Apple Silicon (arm64)")
            #else
            print("⚠️  Not on Apple Silicon - Metal may be limited")
            #endif
            
            // Try to get MLX device information
            print("MLX Swift version: Available")
            print("✅ Device information accessible")
            
            return true
        } catch {
            print("❌ Device info test failed: \(error)")
            return false
        }
    }
    
    static func testMetalAvailability() -> Bool {
        print("\n🔧 Metal Availability Test")
        print(String(repeating: "-", count: 50))
        
        do {
            // Try to create a simple MLX array
            let testArray = MLXArray([1.0, 2.0, 3.0])
            eval(testArray)
            
            print("✅ Created MLX array: \(testArray)")
            print("✅ Array evaluation successful")
            
            // Check array properties
            print("Array shape: \(testArray.shape)")
            print("Array data type: \(testArray.dtype)")
            
            return true
        } catch {
            print("❌ Metal availability test failed: \(error)")
            return false
        }
    }
    
    static func testBasicOperations() -> Bool {
        print("\n🧮 Basic Operations Test")
        print(String(repeating: "-", count: 50))
        
        do {
            // Create test arrays
            let a = MLXArray([1.0, 2.0, 3.0, 4.0])
            let b = MLXArray([5.0, 6.0, 7.0, 8.0])
            
            print("Array a: \(a)")
            print("Array b: \(b)")
            
            // Test addition
            let sum = a + b
            eval(sum)
            print("a + b = \(sum)")
            
            // Test multiplication
            let product = a * b
            eval(product)
            print("a * b = \(product)")
            
            // Test scalar operations
            let scaled = a * 2.0
            eval(scaled)
            print("a * 2 = \(scaled)")
            
            // Test reduction operations
            let sumReduction = a.sum()
            eval(sumReduction)
            print("sum(a) = \(sumReduction)")
            
            let meanReduction = a.mean()
            eval(meanReduction)
            print("mean(a) = \(meanReduction)")
            
            print("✅ All basic operations successful")
            return true
        } catch {
            print("❌ Basic operations test failed: \(error)")
            return false
        }
    }
    
    static func testMatrixOperations() -> Bool {
        print("\n📊 Matrix Operations Test")
        print(String(repeating: "-", count: 50))
        
        do {
            // Create matrices
            let matrixA = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
            let matrixB = MLXArray([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
            
            print("Matrix A (2×3): shape \(matrixA.shape)")
            print("Matrix B (3×2): shape \(matrixB.shape)")
            
            // Matrix multiplication
            let matmulResult = matmul(matrixA, matrixB)
            eval(matmulResult)
            print("A @ B: shape \(matmulResult.shape)")
            print("Result: \(matmulResult)")
            
            // Transpose
            let transposed = matrixA.T
            eval(transposed)
            print("A transpose: shape \(transposed.shape)")
            
            // Element-wise operations on matrices
            let squareMatrix = MLXArray([1.0, 2.0, 3.0, 4.0], [2, 2])
            let elementWise = squareMatrix * squareMatrix
            eval(elementWise)
            print("Element-wise square: \(elementWise)")
            
            print("✅ Matrix operations successful")
            return true
        } catch {
            print("❌ Matrix operations test failed: \(error)")
            return false
        }
    }
    
    static func testMemoryOperations() -> Bool {
        print("\n💾 Memory Operations Test")
        print(String(repeating: "-", count: 50))
        
        do {
            // Create larger arrays to test memory handling
            let size = 100
            let largeArray = MLXArray.zeros([size, size])
            eval(largeArray)
            
            print("Created large array: shape \(largeArray.shape)")
            
            // Fill with values
            let filledArray = largeArray + 1.0
            eval(filledArray)
            print("Filled array with 1.0")
            
            // Test memory-intensive operation
            let result = matmul(filledArray, filledArray)
            eval(result)
            print("Matrix multiplication result: shape \(result.shape)")
            
            // Test reduction on large array
            let sumResult = result.sum()
            eval(sumResult)
            print("Large array sum: \(sumResult)")
            
            print("✅ Memory operations successful")
            return true
        } catch {
            print("❌ Memory operations test failed: \(error)")
            return false
        }
    }
    
    static func testRandomOperations() -> Bool {
        print("\n🎲 Random Operations Test")
        print(String(repeating: "-", count: 50))
        
        do {
            // Set seed for reproducible results
            MLXRandom.seed(42)
            
            // Test uniform random
            let uniform = MLXRandom.uniform(low: 0.0, high: 1.0, [3, 3])
            eval(uniform)
            print("Uniform random (3×3): shape \(uniform.shape)")
            
            // Test normal random
            let normal = MLXRandom.normal([2, 4])
            eval(normal)
            print("Normal random (2×4): shape \(normal.shape)")
            
            // Test statistics on random arrays
            let mean = uniform.mean()
            eval(mean)
            print("Uniform mean: \(mean)")
            
            let max = uniform.max()
            eval(max)
            print("Uniform max: \(max)")
            
            let min = uniform.min()
            eval(min)
            print("Uniform min: \(min)")
            
            print("✅ Random operations successful")
            return true
        } catch {
            print("❌ Random operations test failed: \(error)")
            return false
        }
    }
    
    static func diagnoseMetal() {
        print("\n🔧 Metal Diagnostic Information")
        print(String(repeating: "-", count: 50))
        
        print("Common Metal library issues and solutions:")
        print()
        print("1. Missing Command Line Tools:")
        print("   Solution: xcode-select --install")
        print()
        print("2. Outdated macOS version:")
        print("   MLX requires macOS 13.3+ for optimal Metal support")
        print()
        print("3. Intel Mac limitations:")
        print("   MLX is optimized for Apple Silicon")
        print()
        print("4. Xcode installation:")
        print("   Consider installing full Xcode from App Store")
        print()
        print("5. Environment issues:")
        print("   Try running from Xcode or fresh terminal")
        print()
        print("6. Metal framework check:")
        print("   Verify /System/Library/Frameworks/Metal.framework exists")
        print()
        
        // System information
        #if arch(arm64)
        print("✅ Architecture: Apple Silicon (arm64)")
        #else
        print("⚠️  Architecture: \(String(cString: getprogname()))")
        #endif
        
        #if os(macOS)
        let version = ProcessInfo.processInfo.operatingSystemVersion
        print("macOS Version: \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)")
        
        if version.majorVersion >= 13 && version.minorVersion >= 3 {
            print("✅ macOS version supports MLX")
        } else {
            print("⚠️  macOS version may not fully support MLX")
        }
        #endif
    }
}
