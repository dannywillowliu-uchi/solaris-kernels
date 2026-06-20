// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SolarisVaeMetalDecoder",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "SolarisVaeMetalDecoder", targets: ["SolarisVaeMetalDecoder"]),
        .executable(name: "solaris-vae-metal", targets: ["SolarisVaeMetalCLI"]),
        .executable(name: "quant-gemm-probe", targets: ["QuantGemmProbe"]),
    ],
    targets: [
        .target(
            name: "SolarisVaeMetalDecoder",
            resources: [.process("Kernels")],
            linkerSettings: [
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),
        .executableTarget(
            name: "SolarisVaeMetalCLI",
            dependencies: ["SolarisVaeMetalDecoder"]
        ),
        .executableTarget(
            name: "QuantGemmProbe",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ]
        ),
    ]
)
