#!/bin/bash
export DYLD_FRAMEWORK_PATH="/System/Library/Frameworks:$DYLD_FRAMEWORK_PATH"
export DYLD_LIBRARY_PATH="/usr/lib:$DYLD_LIBRARY_PATH"
export METAL_DEVICE_WRAPPER_TYPE=""
export MLX_METAL_PATH="/System/Library/Frameworks/Metal.framework/Versions/A/Resources"
echo "✅ Metal environment variables set"
echo "DYLD_FRAMEWORK_PATH: $DYLD_FRAMEWORK_PATH"
echo "MLX_METAL_PATH: $MLX_METAL_PATH"
