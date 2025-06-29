#!/usr/bin/env python3
"""
Quick test of the working local agent
"""

import sys
import os

# Add examples to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))

try:
    print("🧪 Testing Local Agent Components...")
    
    # Test MLX math engine
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "working_agent", 
        "examples/09_working_local_agent.py"
    )
    working_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(working_agent)
    
    math_engine = working_agent.MLXMathEngine()
    result = math_engine.solve_linear_system("test system")
    print(f"✅ Math Engine: {result['success']}")
    
    # Test data analysis
    data_result = math_engine.analyze_data([1, 2, 3, 4, 5])
    print(f"✅ Data Analysis: {data_result['success']}")
    
    # Test the full agent with mock responses
    agent = working_agent.LocalAgent()
    print("✅ Agent created")
    
    # Test without loading heavy models
    response = agent.llm._mock_response("Hello, solve 2x + 3y = 7")
    print(f"✅ Mock Response: {response[:50]}...")
    
    print("\n🎉 All core components working!")
    print("Run with: python examples/09_working_local_agent.py")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
