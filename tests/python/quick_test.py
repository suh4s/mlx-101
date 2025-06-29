#!/usr/bin/env python3
"""Quick test of MLX math functionality"""

import sys
sys.path.append('.')
from examples import working_local_agent_09 as agent


def main():
    print("🔢 Quick MLX Math Test:")
    math_tool = MLXMathTool()
    result = math_tool.solve_linear_system("2x + 3y = 13 and x - y = 1")
    print(f"✅ Result: {result}")
    
    # Test matrix operations
    A = mx.array([[1.0, 2.0], [3.0, 4.0]])
    det = mx.linalg.det(A)
    print(f"🔢 Matrix determinant: {det}")

if __name__ == "__main__":
    main()
