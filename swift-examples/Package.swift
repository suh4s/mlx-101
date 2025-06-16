// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MLXSwiftExamples",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .executable(name: "basic-operations", targets: ["BasicOperations"]),
        .executable(name: "linear-algebra", targets: ["LinearAlgebra"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.13.0")
    ],
    targets: [
        .executableTarget(
            name: "BasicOperations",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/BasicOperations"
        ),
        .executableTarget(
            name: "LinearAlgebra",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/LinearAlgebra"
        ),
    ]
)
