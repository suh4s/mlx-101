#!/usr/bin/env python3
"""
Quick test for the comprehensive demo
"""

try:
    import mlx.core as mx
    import numpy as np
    print("✅ MLX and numpy work")
    
    # Test basic MLX operation
    A = mx.array([[2.0, 3.0], [1.0, -1.0]], dtype=mx.float32)
    b = mx.array([7.0, 1.0], dtype=mx.float32)
    
    with mx.stream(mx.cpu):
        solution = mx.linalg.solve(A, b)
        mx.eval(solution)
    
    print(f"✅ MLX linear algebra: x={float(solution[0]):.3f}, y={float(solution[1]):.3f}")
    
    # Test sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence transformers available")
    except ImportError:
        print("❌ Sentence transformers not available")
    
    # Test LangGraph
    try:
        from langgraph.graph import StateGraph, END
        print("✅ LangGraph available")
    except ImportError:
        print("❌ LangGraph not available")
        
    # Test MLX-LM
    try:
        from mlx_lm import load, generate
        print("✅ MLX-LM available")
    except ImportError:
        print("❌ MLX-LM not available")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
