import MLX
import MLXRandom
import Foundation

@main
struct BasicOperations {
    static func main() {
        print("🍎 MLX Swift Basic Operations")
        print("Apple Silicon Machine Learning with Swift")
        print("=" + String(repeating: "=", count: 50))
        
        arrayCreationExamples()
        basicMathOperations()
        arrayManipulation()
        randomOperations()
        interoperabilityDemo()
        
        print("\n" + String(repeating: "=", count: 50))
        print("✅ Swift MLX demo completed successfully!")
        print("Ready to build ML apps with Swift + MLX!")
    }
    
    static func arrayCreationExamples() {
        print("\n1. 🎯 Array Creation")
        print(String(repeating: "-", count: 30))
        
        // Create arrays from Swift arrays
        let arr1 = MLXArray([1, 2, 3, 4, 5])
        print("From Swift array: shape \(arr1.shape)")
        
        // Create with specific shapes
        let zeros = MLXArray.zeros([3, 4])
        let ones = MLXArray.ones([2, 3])
        
        print("Zeros (3×4): \(zeros.shape)")
        print("Ones (2×3): \(ones.shape)")
        
        // Identity matrix
        let identity = MLXArray.eye(3)
        print("Identity (3×3): \(identity.shape)")
    }
    
    static func basicMathOperations() {
        print("\n2. 🧮 Mathematical Operations")
        print(String(repeating: "-", count: 30))
        
        let a = MLXArray([1, 2, 3, 4])
        let b = MLXArray([5, 6, 7, 8])
        
        print("Array a: shape \(a.shape)")
        print("Array b: shape \(b.shape)")
        
        // Element-wise operations
        let sum = a + b
        let product = a * b
        let power = pow(a, 2)
        
        print("a + b: shape \(sum.shape)")
        print("a * b: shape \(product.shape)")
        print("a²: shape \(power.shape)")
        
        // Scalar operations
        let scalarAdd = a + 10
        let scalarMul = a * 3
        
        print("a + 10: shape \(scalarAdd.shape)")
        print("a * 3: shape \(scalarMul.shape)")
    }
    
    static func arrayManipulation() {
        print("\n3. 📐 Array Manipulation")
        print(String(repeating: "-", count: 30))
        
        // Create matrix using flat array with shape
        let flatData = [1, 2, 3, 4, 5, 6]
        let matrix = MLXArray(flatData, [2, 3])
        print("Original matrix (2×3): \(matrix.shape)")
        
        // Transpose
        let transposed = matrix.T
        print("Transposed (3×2): \(transposed.shape)")
        
        // Reshaping
        let reshaped = matrix.reshaped([3, 2])
        print("Reshaped (3×2): \(reshaped.shape)")
        
        // Slicing
        let firstRow = matrix[0]
        print("First row: shape \(firstRow.shape)")
        
        // Reductions
        let sum = matrix.sum()
        let mean = matrix.mean()
        let max = matrix.max()
        
        print("Sum: \(sum.item(Int.self))")
        print("Mean: \(mean.item(Float.self))")
        print("Max: \(max.item(Int.self))")
    }
    
    static func randomOperations() {
        print("\n4. 🎲 Random Operations")
        print(String(repeating: "-", count: 30))
        
        // Set random seed for reproducibility
        MLXRandom.seed(42)
        
        // Generate random arrays
        let uniform = MLXRandom.uniform(low: 0, high: 1, [3, 3])
        let normal = MLXRandom.normal([2, 4])
        
        print("Uniform random (3×3): \(uniform.shape)")
        print("Normal random (2×4): \(normal.shape)")
        
        // Statistics
        print("Uniform mean: \(uniform.mean().item(Float.self))")
        // Use standard deviation calculation
        let normalMean = normal.mean()
        let variance = ((normal - normalMean) * (normal - normalMean)).mean()
        let std = sqrt(variance)
        print("Normal std: \(std.item(Float.self))")
    }
    
    static func interoperabilityDemo() {
        print("\n5. 🔄 Swift Integration")
        print(String(repeating: "-", count: 30))
        
        // Swift array to MLX
        let swiftArray = [1.0, 2.0, 3.0, 4.0, 5.0]
        let mlxArray = MLXArray(swiftArray)
        
        print("Swift array: \(swiftArray)")
        print("MLX array: \(mlxArray)")
        
        // Perform MLX operations
        let result = sqrt(mlxArray)
        print("sqrt(MLX): \(result)")
        
        // MLX arrays work seamlessly with Swift
        let doubled = mlxArray * 2
        let sum = doubled.sum()
        
        print("Doubled and summed: \(sum)")
        
        // Type safety and Swift features
        print("MLX integrates with Swift's type system")
        print("Shape: \(mlxArray.shape)")
        print("Data type: \(mlxArray.dtype)")
    }
}
