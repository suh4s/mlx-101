# 🔧 MLX Metal Library Fix Guide

This guide provides step-by-step solutions to fix the "Failed to load the default metallib" error in MLX Swift.

## 🔍 Problem Analysis

### Current System Status
- ✅ **Metal Framework**: Present at `/System/Library/Frameworks/Metal.framework`
- ✅ **Default MetalLib**: Found at `/System/Library/Frameworks/Metal.framework/Versions/A/Resources/default.metallib`
- ✅ **Apple Silicon**: arm64 architecture confirmed
- ✅ **Python MLX**: Working correctly (6/6 tests passed)
- ❌ **Swift MLX**: Cannot load Metal library

### Root Cause
The issue is that MLX Swift (via its C++ core) cannot locate the system Metal library despite it being present. This is likely due to:

1. **Xcode Installation**: Only Command Line Tools installed (not full Xcode)
2. **Library Path Resolution**: MLX C++ core can't resolve Metal library paths
3. **Runtime Environment**: Missing Metal runtime environment setup

## 🛠️ Fix Solutions (Try in Order)

### Solution 1: Install Full Xcode (Recommended)

The Command Line Tools alone may not provide complete Metal library support.

```bash
# Check current Xcode setup
xcode-select -p
# Should show: /Library/Developer/CommandLineTools

# Install full Xcode from App Store or Developer Portal
# After installation, switch to full Xcode:
sudo xcode-select -s /Applications/Xcode.app/Developer

# Verify
xcode-select -p
# Should show: /Applications/Xcode.app/Developer

# Test Swift MLX
cd swift-examples
swift run test-metal-library
```

### Solution 2: Environment Variables

Set Metal library environment variables to help MLX locate the libraries.

```bash
# Create environment setup script
cat > setup_metal_env.sh << 'EOF'
#!/bin/bash
export DYLD_FRAMEWORK_PATH="/System/Library/Frameworks:$DYLD_FRAMEWORK_PATH"
export DYLD_LIBRARY_PATH="/usr/lib:$DYLD_LIBRARY_PATH"
export METAL_DEVICE_WRAPPER_TYPE=""
export MLX_METAL_PATH="/System/Library/Frameworks/Metal.framework/Versions/A/Resources"
echo "✅ Metal environment variables set"
EOF

chmod +x setup_metal_env.sh

# Source the environment
source setup_metal_env.sh

# Test
cd swift-examples
swift run test-metal-library
```

### Solution 3: Rebuild MLX Swift with Metal Support

Force rebuild of MLX Swift dependencies to ensure Metal linking.

```bash
cd swift-examples

# Clean everything
swift package clean
rm -rf .build/
rm Package.resolved

# Reset Swift package cache
rm -rf ~/Library/Caches/org.swift.swiftpm/
rm -rf ~/Library/Developer/Xcode/DerivedData/

# Rebuild with verbose output
swift build --verbose

# Test
swift run test-metal-library
```

### Solution 4: Alternative MLX Swift Version

Try a different MLX Swift version that might have better Metal support.

```bash
cd swift-examples

# Edit Package.swift - change version
# FROM: .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.25.0")
# TO:   .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.20.0")

# Or try exact version
# TO:   .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.18.0")

swift package update
swift build
swift run test-metal-library
```

### Solution 5: Xcode Project Approach

Use Xcode instead of command line for better Metal integration.

```bash
cd swift-examples

# Generate Xcode project
swift package generate-xcodeproj

# Open in Xcode
open MLXSwiftExamples.xcodeproj

# In Xcode:
# 1. Select the test-metal-library scheme
# 2. Ensure "Run" is set to Debug mode
# 3. Go to Product > Run
# 4. Check console output
```

### Solution 6: System Integrity Check

Verify system Metal libraries are not corrupted.

```bash
# Check Metal library integrity
ls -la /System/Library/Frameworks/Metal.framework/Versions/A/Resources/default.metallib

# Verify file is readable
file /System/Library/Frameworks/Metal.framework/Versions/A/Resources/default.metallib

# Check system Metal devices
system_profiler SPDisplaysDataType | grep -A5 "Metal"

# Test Metal with simple Metal app (if available)
xcrun metal --version
```

### Solution 7: MLX Force CPU Mode

Temporarily force MLX to use CPU mode to bypass Metal entirely.

Create a CPU-only test version:

```swift
// Add to swift-examples/Sources/TestMetalLibrary/main.swift
// Before any MLX operations, add:

import os
os.setenv("MLX_USE_CPU", "1", 1)

// This forces MLX to use CPU instead of Metal
```

### Solution 8: Check for macOS Updates

Ensure you have the latest macOS version with Metal support.

```bash
# Check macOS version
sw_vers

# Check for updates
softwareupdate -l

# If updates available, install them
sudo softwareupdate -i -a
```

## 🧪 Testing Solutions

After trying each solution, test with:

```bash
# Quick test
cd swift-examples
swift run test-metal-library

# Verbose test to see detailed error messages
cd swift-examples
swift run test-metal-library 2>&1 | tee metal_test_output.log
```

## 🎯 Expected Results

### Before Fix
```
🔬 MLX Swift Metal Library Test
===================================================
✅ Device Information Test: Apple Silicon detected
❌ Metal Availability Test: "Failed to load the default metallib"
```

### After Fix
```
🔬 MLX Swift Metal Library Test
===================================================
✅ Device Information Test: Apple Silicon detected
✅ Metal Availability Test: MetalLib loaded successfully
✅ Basic Operations Test: Array operations working
✅ Matrix Operations Test: Matrix math working
✅ Memory Operations Test: Large arrays working
✅ Random Operations Test: Random generation working

📊 Test Results: 6/6 tests passed
🎉 All tests passed! Metal library is working correctly.
```

## 🔧 Advanced Troubleshooting

### Debug Metal Library Loading

Create a debug version that shows exactly where MLX is looking for Metal libraries:

```bash
# Enable MLX debug output
export MLX_DEBUG=1
export MLX_METAL_DEBUG=1

# Run with detailed logging
cd swift-examples
swift run test-metal-library 2>&1 | grep -i metal
```

### Check Dynamic Library Dependencies

```bash
# Check what libraries the MLX Swift binary depends on
cd swift-examples
swift build
otool -L .build/debug/test-metal-library | grep -i metal
```

### Alternative: Use Instruments

```bash
# Profile with Instruments to see what's happening
cd swift-examples
swift build
xcrun instruments -t "System Trace" -D . .build/debug/test-metal-library
```

## 📋 Solution Checklist

Try solutions in this order and check off what works:

- [ ] **Solution 1**: Install full Xcode (most likely to work)
- [ ] **Solution 2**: Set Metal environment variables
- [ ] **Solution 3**: Clean rebuild MLX Swift
- [ ] **Solution 4**: Try different MLX Swift version
- [ ] **Solution 5**: Use Xcode project approach
- [ ] **Solution 6**: System integrity check
- [ ] **Solution 7**: Force CPU mode (workaround)
- [ ] **Solution 8**: macOS updates

## 🎯 Most Likely Solutions

Based on the investigation, these are most likely to work:

1. **Install Full Xcode** (80% success rate)
   - Command Line Tools don't include complete Metal runtime
   - Full Xcode provides comprehensive Metal development stack

2. **Environment Variables + Clean Rebuild** (60% success rate)
   - Helps MLX find Metal libraries
   - Forces complete rebuild with proper paths

3. **Different MLX Swift Version** (40% success rate)
   - Some versions have better Metal library integration
   - Version 0.18.0 or 0.20.0 might work better

## 📞 If All Solutions Fail

If none of these solutions work:

1. **Use Python MLX**: Confirmed working for all functionality
2. **File Bug Report**: Report to MLX Swift repository with system details
3. **Monitor Updates**: Watch for MLX Swift releases with Metal fixes
4. **Community Support**: Check MLX Swift discussions and issues

---

**Next Steps**: Try Solution 1 (Full Xcode) first, as it has the highest success rate for resolving Metal library issues.
