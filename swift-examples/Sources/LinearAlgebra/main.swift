import MLX
import MLXRandom
import Foundation

@main
struct LinearAlgebra {
    static func main() {
        print("🧮 MLX Swift Linear Algebra")
        print("Advanced Mathematical Operations with Swift")
        print("=" + String(repeating: "=", count: 50))
        
        matrixOperations()
        linearSystems()
        decompositions()
        neuralNetworkBasics()
        
        print("\n" + String(repeating: "=", count: 50))
        print("✅ Linear algebra demo completed!")
        print("Ready for advanced ML with Swift + MLX!")
    }
    
    static func matrixOperations() {
        print("\n1. 📊 Matrix Operations")
        print(String(repeating: "-", count: 30))
        
        let A = MLXArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        let B = MLXArray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        
        print("Matrix A (2×3): \(A.shape)")
        print("Matrix B (3×2): \(B.shape)")
        
        // Matrix multiplication
        let C = matmul(A, B)
        print("A @ B (2×2): \(C.shape)")
        print("Result: \(C)")
        
        // Element-wise operations
        let squareMatrix = MLXArray([[1.0, 2.0], [3.0, 4.0]])
        let elementWise = squareMatrix * squareMatrix
        print("Element-wise multiplication: \(elementWise)")
        
        // Transpose
        print("A transpose (3×2): \(A.T.shape)")
    }
    
    static func linearSystems() {
        print("\n2. ⚖️ Linear Systems")
        print(String(repeating: "-", count: 30))
        
        // Solve Ax = b
        let A = MLXArray([[2.0, 1.0, -1.0],
                          [-3.0, -1.0, 2.0],
                          [-2.0, 1.0, 2.0]])
        let b = MLXArray([8.0, -11.0, -3.0])
        
        print("Coefficient matrix A (3×3): \(A.shape)")
        print("Constants vector b (3): \(b.shape)")
        
        // Solve the system
        let x = MLXArray.solve(A, b)
        print("Solution x: \(x)")
        
        // Verify solution
        let verification = matmul(A, x)
        print("Verification A*x: \(verification)")
        print("Original b: \(b)")
    }
    
    static func decompositions() {
        print("\n3. 🔬 Matrix Decompositions")
        print(String(repeating: "-", count: 30))
        
        let matrix = MLXArray([[4.0, 2.0, 1.0],
                               [2.0, 3.0, 2.0],
                               [1.0, 2.0, 5.0]])
        
        print("Matrix (3×3): \(matrix.shape)")
        
        // QR Decomposition
        let (Q, R) = MLXArray.qr(matrix)
        print("QR Decomposition:")
        print("Q (orthogonal): \(Q.shape)")
        print("R (upper triangular): \(R.shape)")
        
        // Verify reconstruction
        let reconstruction = matmul(Q, R)
        print("Q @ R (should equal original): \(reconstruction.shape)")
        
        // SVD Decomposition
        let (U, s, Vt) = MLXArray.svd(matrix)
        print("SVD Decomposition:")
        print("U: \(U.shape)")
        print("s (singular values): \(s.shape)")
        print("Vt: \(Vt.shape)")
        
        // Matrix inverse
        do {
            let inverse = MLXArray.inverse(matrix)
            print("Matrix inverse: \(inverse.shape)")
            
            // Verify A * A^(-1) ≈ I
            let identity = matmul(matrix, inverse)
            print("A * A^(-1) (should be identity): \(identity.shape)")
        } catch {
            print("Matrix inversion failed: \(error)")
        }
    }
    
    static func neuralNetworkBasics() {
        print("\n4. 🧠 Neural Network Basics")
        print(String(repeating: "-", count: 30))
        
        // Simulate a simple neural network layer
        let batchSize = 4
        let inputFeatures = 3
        let outputFeatures = 2
        
        // Input data
        MLXRandom.seed(42)
        let input = MLXRandom.normal([batchSize, inputFeatures])
        
        // Layer parameters
        let weights = MLXRandom.normal([inputFeatures, outputFeatures]) * 0.1
        let bias = MLXArray.zeros([outputFeatures])
        
        print("Input shape: \(input.shape)")
        print("Weights shape: \(weights.shape)")
        print("Bias shape: \(bias.shape)")
        
        // Forward pass
        let linear = matmul(input, weights) + bias
        let activated = maximum(linear, MLXArray.zeros(linear.shape)) // ReLU
        
        print("Linear output shape: \(linear.shape)")
        print("After ReLU: \(activated.shape)")
        print("Sample output: \(activated)")
        
        // Batch operations
        let batchSum = activated.sum(axis: 0)
        let batchMean = activated.mean(axis: 0)
        
        print("Batch sum: \(batchSum)")
        print("Batch mean: \(batchMean)")
        
        // Demonstrate broadcasting
        let scaledInput = input * 2.0
        let normalized = (input - input.mean()) / input.standardDeviation()
        
        print("Scaled input shape: \(scaledInput.shape)")
        print("Normalized input mean: \(normalized.mean())")
        print("Normalized input std: \(normalized.standardDeviation())")
    }
}
