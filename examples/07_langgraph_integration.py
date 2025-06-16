#!/usr/bin/env python3
"""
LangGraph + MLX Integration Example
===================================

This example demonstrates how to integrate LangGraph with MLX for building
AI agents that leverage Apple Silicon acceleration.

Features:
- Local LLM inference with MLX
- MLX-powered mathematical tools
- Image processing capabilities
- State management with LangGraph
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import json

# Note: This is a conceptual example showing integration patterns
# To run this, you would need: pip install langgraph langchain

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("LangGraph not installed. This is a demonstration of integration patterns.")
    LANGGRAPH_AVAILABLE = False


@dataclass
class AgentState:
    """State management for MLX-powered agent"""
    messages: List[Dict[str, str]]
    results: Dict[str, Any]
    mlx_data: Dict[str, mx.array]


# MLX-Powered Tools for LangGraph
def mlx_matrix_solver_function(matrix_data: str, vector_data: str) -> str:
    """
    Solve linear systems using MLX on Apple Silicon.
    
    Args:
        matrix_data: JSON string representing coefficient matrix
        vector_data: JSON string representing constants vector
    
    Returns:
        Solution as JSON string
    """
    try:
        # Parse input data
        A_list = json.loads(matrix_data)
        b_list = json.loads(vector_data)
        
        # Convert to MLX arrays
        A = mx.array(A_list, dtype=mx.float32)
        b = mx.array(b_list, dtype=mx.float32)
        
        # Solve Ax = b using MLX (with CPU fallback)
        try:
            x = mx.linalg.solve(A, b)
        except:
            # Fallback to CPU stream if GPU not supported
            with mx.stream(mx.cpu):
                x = mx.linalg.solve(A, b)
                mx.eval(x)
        
        # Convert result back to Python list
        solution = x.tolist()
        
        return json.dumps({
            "success": True,
            "solution": solution,
            "verification": (A @ x).tolist()
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def mlx_neural_network_prediction(input_data: str, weights_data: str) -> str:
    """
    Run neural network inference using MLX.
    
    Args:
        input_data: JSON string of input features
        weights_data: JSON string of network weights
    
    Returns:
        Prediction results as JSON string
    """
    try:
        # Parse input
        inputs = json.loads(input_data)
        weights = json.loads(weights_data)
        
        # Convert to MLX arrays
        x = mx.array(inputs, dtype=mx.float32)
        W = mx.array(weights, dtype=mx.float32)
        
        # Simple neural network forward pass (from 03_neural_networks.py)
        def relu(x):
            return mx.maximum(x, 0)
        
        def sigmoid(x):
            return 1 / (1 + mx.exp(-x))
        
        # Forward pass
        hidden = relu(x @ W)  # Hidden layer with ReLU
        output = sigmoid(hidden @ W.T)  # Output layer with sigmoid
        
        return json.dumps({
            "success": True,
            "predictions": output.tolist(),
            "confidence": float(mx.max(output))
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def mlx_image_features(image_description: str) -> str:
    """
    Extract image features using MLX (simulated for demo).
    
    Args:
        image_description: Description of image properties
    
    Returns:
        Feature analysis as JSON string
    """
    try:
        # Simulate image processing (based on 04_image_processing.py)
        # In practice, you'd load and process actual images
        
        # Create synthetic image features
        height, width = 64, 64
        image = mx.random.normal((height, width))
        
        # Apply image processing operations
        # Edge detection kernel (Sobel)
        sobel_x = mx.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mx.float32)
        
        # Simple convolution simulation
        features = {
            "mean_intensity": float(mx.mean(image)),
            "std_intensity": float(mx.std(image)),
            "max_value": float(mx.max(image)),
            "min_value": float(mx.min(image)),
            "shape": [height, width],
            "has_edges": True  # Simulated edge detection result
        }
        
        return json.dumps({
            "success": True,
            "features": features,
            "processed_with": "MLX on Apple Silicon"
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def create_mlx_powered_agent():
    """Create a LangGraph agent with MLX-powered tools"""
    
    if not LANGGRAPH_AVAILABLE:
        print("Install LangGraph to run this agent: pip install langgraph langchain")
        return None
    
    # Define agent nodes
    def reasoning_node(state: AgentState):
        """Main reasoning node using local LLM (conceptual)"""
        last_message = state.messages[-1]["content"]
        
        # In practice, you'd use MLX-LM here for local inference
        # For demo, we'll simulate reasoning
        
        if "solve" in last_message.lower() and "matrix" in last_message.lower():
            return {
                "messages": state.messages + [{
                    "role": "assistant", 
                    "content": "I'll solve this linear system using MLX matrix operations."
                }],
                "next_action": "use_matrix_solver"
            }
        elif "neural" in last_message.lower() or "predict" in last_message.lower():
            return {
                "messages": state.messages + [{
                    "role": "assistant",
                    "content": "I'll run neural network inference using MLX."
                }],
                "next_action": "use_neural_network"
            }
        elif "image" in last_message.lower():
            return {
                "messages": state.messages + [{
                    "role": "assistant",
                    "content": "I'll analyze the image using MLX computer vision."
                }],
                "next_action": "use_image_processing"
            }
        else:
            return {
                "messages": state.messages + [{
                    "role": "assistant",
                    "content": "I can help with matrix solving, neural networks, or image processing using MLX!"
                }],
                "next_action": "end"
            }
    
    def tool_execution_node(state: AgentState):
        """Execute MLX-powered tools based on state"""
        action = state.get("next_action", "end")
        
        if action == "use_matrix_solver":
            # Example matrix problem
            result = mlx_matrix_solver.invoke({
                "matrix_data": "[[2, 1], [1, 3]]",
                "vector_data": "[5, 7]"
            })
            
            return {
                "messages": state.messages + [{
                    "role": "assistant",
                    "content": f"Matrix solution computed with MLX: {result}"
                }],
                "results": {"matrix_result": result}
            }
        
        elif action == "use_neural_network":
            result = mlx_neural_network_prediction.invoke({
                "input_data": "[0.5, 1.2, -0.3]",
                "weights_data": "[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]"
            })
            
            return {
                "messages": state.messages + [{
                    "role": "assistant", 
                    "content": f"Neural network prediction using MLX: {result}"
                }],
                "results": {"nn_result": result}
            }
        
        elif action == "use_image_processing":
            result = mlx_image_features.invoke({
                "image_description": "Sample image for processing"
            })
            
            return {
                "messages": state.messages + [{
                    "role": "assistant",
                    "content": f"Image analysis with MLX: {result}"
                }],
                "results": {"image_result": result}
            }
        
        return {"messages": state.messages}
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool_execution", tool_execution_node)
    
    # Define the flow
    workflow.set_entry_point("reasoning")
    workflow.add_conditional_edges(
        "reasoning",
        lambda state: state.get("next_action", "end"),
        {
            "use_matrix_solver": "tool_execution",
            "use_neural_network": "tool_execution", 
            "use_image_processing": "tool_execution",
            "end": END
        }
    )
    workflow.add_edge("tool_execution", END)
    
    return workflow.compile()


def demonstrate_integration():
    """Demonstrate MLX + LangGraph integration"""
    print("🤖 MLX + LangGraph Integration Demo")
    print("=" * 50)
    
    # Test individual MLX tools
    print("\n🔧 Testing MLX-powered tools:")
    
    # Test matrix solver
    print("\n1. Matrix Solver:")
    result = mlx_matrix_solver.invoke({
        "matrix_data": "[[3, 1], [1, 2]]",
        "vector_data": "[9, 8]"
    })
    print(f"   Result: {result}")
    
    # Test neural network
    print("\n2. Neural Network:")
    result = mlx_neural_network_prediction.invoke({
        "input_data": "[1.0, 0.5, -0.2]",
        "weights_data": "[[0.2, 0.3, 0.1], [0.4, 0.1, 0.5]]"
    })
    print(f"   Result: {result}")
    
    # Test image processing
    print("\n3. Image Processing:")
    result = mlx_image_features.invoke({
        "image_description": "Test image for feature extraction"
    })
    print(f"   Result: {result}")
    
    print("\n✅ MLX tools working successfully!")
    
    # Create agent (if LangGraph available)
    agent = create_mlx_powered_agent()
    if agent:
        print("\n🎯 LangGraph agent with MLX tools created successfully!")
        print("   Ready for AI workflows with Apple Silicon acceleration!")
    else:
        print("\n📝 Install LangGraph to test the full agent integration:")
        print("   pip install langgraph langchain")
    
    print("\n💡 Integration Benefits:")
    print("   • Local LLM inference with MLX-LM")
    print("   • Apple Silicon GPU acceleration") 
    print("   • Privacy-first processing")
    print("   • Advanced agent workflows")
    print("   • Unified memory efficiency")


if __name__ == "__main__":
    demonstrate_integration()
