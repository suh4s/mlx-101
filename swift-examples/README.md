# 🚀 MLX Swift Examples

This directory contains Swift implementations of MLX-powered applications, demonstrating native Apple Silicon performance for machine learning and mathematical operations.

## 📁 Structure

```
swift-examples/
├── Package.swift              # Swift Package Manager configuration
├── README.md                 # This file
└── Sources/
    ├── BasicOperations/      # Basic MLX operations
    ├── LinearAlgebra/       # Linear algebra examples
    └── LocalMLXAgent/       # Complete local AI agent
```

## 🎯 Examples

### 1. Basic Operations (`basic-operations`)
Fundamental MLX operations including:
- Array creation and manipulation
- Mathematical operations
- Device management

### 2. Linear Algebra (`linear-algebra`)
Advanced mathematical operations:
- Matrix multiplication
- Eigenvalue decomposition
- Linear system solving

### 3. Local MLX Agent (`local-mlx-agent`)
Complete AI agent implementation featuring:
- **Mathematical reasoning** with MLX linear algebra
- **Knowledge search** with simple document matching
- **Interactive processing** of user queries
- **Apple Silicon optimization** for performance

## 🛠️ Prerequisites

- **macOS 13.0+** or **iOS 16.0+**
- **Xcode 15.0+** with Swift 5.9+
- **Apple Silicon Mac** (recommended for optimal performance)

## 🚀 Quick Start

### 1. Build All Examples
```bash
cd swift-examples
swift build
```

### 2. Run the Local Agent Demo
```bash
swift run local-mlx-agent
```

### 3. Test Linear Algebra Operations
```bash
swift run linear-algebra
```

### 4. Try Basic Operations
```bash
swift run basic-operations
```

## 📊 Expected Output

### Local MLX Agent Demo:
```
🚀 MLX Swift Local Agent Demo
==================================================
🍎 Running natively on Apple Silicon
🧠 Using MLX Swift for mathematical operations
📚 Local knowledge base for information retrieval

🤖 Processing: 'Solve the linear equation 2x + 3y = 7 and x - y = 1'
--------------------------------------------------
🔢 Math Solution: x = 2.0, y = 1.0
✅ Query processed using MLX on Apple Silicon

🔢 MLX Matrix Operations Demo
========================================
Matrix A (3x3):
[random matrix values]

Eigenvalues of A:
[eigenvalue results]

✨ Demo completed successfully!
💡 This demonstrates local AI processing using MLX Swift
```

## 🏗️ Architecture

The Local MLX Agent demonstrates a complete AI processing pipeline:

```
┌─────────────────────────────────────────┐
│            USER QUERY                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           LOCAL AGENT                   │
│  ┌─────────────┬─────────────────────┐  │
│  │ MATH ENGINE │  KNOWLEDGE BASE     │  │
│  │             │                     │  │
│  │ • MLX       │ • Document Search   │  │
│  │   Algebra   │ • Keyword Matching  │  │
│  │ • Matrix    │ • Information       │  │
│  │   Ops       │   Retrieval         │  │
│  └─────────────┴─────────────────────┘  │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         PROCESSED RESPONSE              │
│   • Mathematical Results               │
│   • Knowledge Answers                  │
│   • Apple Silicon Performance          │
└─────────────────────────────────────────┘
```

## 🧪 Key Features

### Performance
- **Native Apple Silicon** optimization
- **Hardware-accelerated** matrix operations
- **Memory-efficient** computation with MLX

### Privacy
- **100% local processing** - no external API calls
- **On-device computation** for sensitive data
- **Offline capability** for disconnected environments

### Scalability
- **Modular architecture** for easy extension
- **Type-safe Swift** for robust development
- **Package Manager** integration for dependencies

## 🔗 Integration with Python Examples

These Swift examples complement the Python implementations in `../examples/`:

| Swift Example | Python Equivalent | Focus |
|---------------|-------------------|-------|
| `local-mlx-agent` | `09_working_local_agent.py` | Complete agent |
| `linear-algebra` | `02_linear_algebra.py` | Math operations |
| `basic-operations` | `01_basic_operations.py` | MLX fundamentals |

## 📚 Learning Path

1. **Start with** `basic-operations` to understand MLX Swift basics
2. **Progress to** `linear-algebra` for mathematical operations
3. **Explore** `local-mlx-agent` for complete AI agent patterns
4. **Compare with** Python examples for language differences
5. **Experiment with** the Jupyter notebook in `../notebooks/`

## 🛠️ Development Tips

### Building for Different Targets
```bash
# Build for current platform
swift build

# Build for release (optimized)
swift build -c release

# Build specific target
swift build --target LocalMLXAgent
```

### Debugging
```bash
# Run with debug information
swift run -c debug local-mlx-agent

# Use Xcode for advanced debugging
open Package.swift
```

### Performance Profiling
- Use Xcode Instruments for performance analysis
- Monitor memory usage with Activity Monitor
- Compare performance between debug and release builds

## 🤝 Contributing

To add new Swift examples:

1. Create a new directory in `Sources/`
2. Add your Swift files with a `main.swift` entry point
3. Update `Package.swift` to include the new target
4. Add documentation and examples
5. Test on both Apple Silicon and Intel Macs (if available)

## 📖 Resources

- [MLX Swift Documentation](https://github.com/ml-explore/mlx-swift)
- [Apple Silicon Optimization Guide](https://developer.apple.com/documentation/accelerate)
- [Swift Package Manager](https://swift.org/package-manager/)
- [MLX Framework Overview](https://github.com/ml-explore/mlx)
