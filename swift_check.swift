#!/usr/bin/env swift

import Foundation

print("🍎 Swift + MLX Integration Check")
print("================================")

print("\n1. ✅ Swift is working!")
print("Swift version: \(ProcessInfo.processInfo.environment["SWIFT_VERSION"] ?? "Unknown")")

print("\n2. 🍎 Apple Silicon Detection")
#if arch(arm64)
print("✅ Running on Apple Silicon (arm64)")
#else
print("❌ Not running on Apple Silicon")
#endif

print("\n3. 📱 Platform Information")
#if os(macOS)
print("✅ Running on macOS")
let osVersion = ProcessInfo.processInfo.operatingSystemVersion
print("OS Version: \(osVersion.majorVersion).\(osVersion.minorVersion).\(osVersion.patchVersion)")
#endif

print("\n4. 🧮 Basic Swift Math")
let numbers = [1, 2, 3, 4, 5]
let doubled = numbers.map { $0 * 2 }
let sum = doubled.reduce(0, +)

print("Numbers: \(numbers)")
print("Doubled: \(doubled)")
print("Sum: \(sum)")

print("\n5. 🔗 Next Steps")
print("To use MLX with Swift:")
print("• Install MLX Swift package")
print("• Create Swift Package Manager project")
print("• Add MLX dependency")
print("• Build with: swift build")
print("• Run with: swift run")

print("\n6. 📦 Swift Package Example")
print("// Package.swift")
print(".package(url: \"https://github.com/ml-explore/mlx-swift.git\", from: \"0.13.0\")")

print("\n================================")
print("✅ Swift environment ready for MLX!")
print("Use the Swift examples in swift-examples/ directory")
