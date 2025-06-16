#!/usr/bin/env python3
"""
🚀 LangGraph + MLX Demo: Smart Math Assistant
=============================================

A practical demonstration of LangGraph + MLX integration.
This creates a smart assistant that combines:

• 🧠 LangGraph: Agent workflows and state management
• ⚡ MLX: Apple Silicon mathematical computation
• 🔒 Local Processing: Everything runs on-device

Features:
- Solve linear systems with MLX
- Statistical analysis with Apple Silicon
- Agent-based task orchestration
- Interactive command-line interface

To test without local LLM:
python examples/05_interactive_langgraph_mlx_demo.py
"""

import mlx.core as mx
import time
from typing import Dict, List, Any, TypedDict

# Check for LangGraph (graceful degradation if not available)
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph available")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not available - running in standalone mode")


class MathAgentState(TypedDict):
    """State for our mathematical agent"""
    messages: List[Dict[str, str]]  # Simplified for demo
    current_problem: str
    solution: Dict[str, Any]
    step_count: int


class MLXMathProcessor:
    """MLX-powered mathematical computation engine"""
    
    @staticmethod
    def solve_linear_system(equations: str) -> Dict[str, Any]:
        """
        Solve linear systems using MLX
        
        For demo, we'll solve: 2x + y = 5, x + 3y = 7
        """
        try:
            # Example system: 2x + y = 5, x + 3y = 7
            A = mx.array([[2, 1], [1, 3]], dtype=mx.float32)
            b = mx.array([5, 7], dtype=mx.float32)
            
            # Solve using MLX with CPU stream for compatibility
            start_time = time.time()
            with mx.stream(mx.cpu):
                x = mx.linalg.solve(A, b)
                mx.eval(x)
            solve_time = time.time() - start_time
            
            # Verification
            verification = A @ x
            
            return {
                "success": True,
                "solution": [float(x[0]), float(x[1])],
                "verification": [float(verification[0]), float(verification[1])],
                "solve_time": solve_time,
                "method": "MLX Linear Solver on Apple Silicon"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def matrix_analysis(description: str) -> Dict[str, Any]:
        """Analyze matrix properties using MLX"""
        try:
            # Example matrix for demo
            matrix = mx.array([[4, 2, 1], [2, 3, 2], [1, 2, 5]], dtype=mx.float32)
            
            start_time = time.time()
            
            # Compute properties
            trace_val = float(mx.trace(matrix))
            norm_val = float(mx.linalg.norm(matrix))
            
            # Try determinant (may need CPU stream)
            try:
                det_val = float(mx.linalg.det(matrix))
            except:
                with mx.stream(mx.cpu):
                    det_val = float(mx.linalg.det(matrix))
            
            compute_time = time.time() - start_time
            
            return {
                "success": True,
                "matrix_shape": list(matrix.shape),
                "trace": trace_val,
                "frobenius_norm": norm_val,
                "determinant": det_val,
                "compute_time": compute_time,
                "platform": "MLX on Apple Silicon"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def statistical_analysis(data_description: str) -> Dict[str, Any]:
        """Perform statistical analysis using MLX vectorized operations"""
        try:
            # Sample data for demo
            data = [2.1, 3.5, 2.8, 4.2, 3.1, 2.9, 3.7, 2.4, 3.8, 3.3, 
                   2.7, 3.4, 2.2, 3.9, 3.0, 2.6, 3.6, 2.5, 3.2, 2.8]
            
            # Convert to MLX array for vectorized computation
            arr = mx.array(data, dtype=mx.float32)
            
            start_time = time.time()
            
            # Compute statistics using MLX
            mean_val = float(mx.mean(arr))
            std_val = float(mx.std(arr))
            min_val = float(mx.min(arr))
            max_val = float(mx.max(arr))
            sum_val = float(mx.sum(arr))
            
            # Sorted array for percentiles
            sorted_arr = mx.sort(arr)
            n = len(data)
            median_val = float(sorted_arr[n//2])
            
            compute_time = time.time() - start_time
            
            return {
                "success": True,
                "count": len(data),
                "mean": mean_val,
                "std_dev": std_val,
                "min": min_val,
                "max": max_val,
                "sum": sum_val,
                "median": median_val,
                "range": max_val - min_val,
                "compute_time": compute_time,
                "method": "MLX vectorized operations"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def create_simple_agent():
    """Create a simple math agent using LangGraph"""
    
    if not LANGGRAPH_AVAILABLE:
        return None
    
    math_processor = MLXMathProcessor()
    
    def planning_node(state: MathAgentState) -> MathAgentState:
        """Plan the mathematical approach"""
        problem = state["current_problem"].lower()
        step = state.get("step_count", 0) + 1
        
        if "linear" in problem or "system" in problem or "equation" in problem:
            approach = "linear_system"
            plan = f"📋 Step {step}: I'll solve this linear system using MLX linear algebra"
        elif "matrix" in problem:
            approach = "matrix_analysis"  
            plan = f"📋 Step {step}: I'll analyze matrix properties using MLX"
        elif "statistics" in problem or "data" in problem:
            approach = "statistics"
            plan = f"📋 Step {step}: I'll perform statistical analysis using MLX"
        else:
            approach = "general"
            plan = f"📋 Step {step}: I'll analyze this mathematically using MLX"
        
        return {
            **state,
            "step_count": step,
            "messages": state["messages"] + [{"role": "assistant", "content": plan}],
            "solution": {"approach": approach}
        }
    
    def computation_node(state: MathAgentState) -> MathAgentState:
        """Perform the mathematical computation"""
        approach = state["solution"].get("approach", "general")
        
        if approach == "linear_system":
            result = math_processor.solve_linear_system(state["current_problem"])
        elif approach == "matrix_analysis":
            result = math_processor.matrix_analysis(state["current_problem"])
        elif approach == "statistics":
            result = math_processor.statistical_analysis(state["current_problem"])
        else:
            # Default to statistics for demo
            result = math_processor.statistical_analysis(state["current_problem"])
        
        if result["success"]:
            response = format_math_result(result, approach)
        else:
            response = f"❌ Computation error: {result['error']}"
        
        return {
            **state,
            "solution": {**state["solution"], **result},
            "messages": state["messages"] + [{"role": "assistant", "content": response}]
        }
    
    def summary_node(state: MathAgentState) -> MathAgentState:
        """Provide final summary"""
        summary = f"""
📊 **Computation Summary**:
- Problem type: {state['solution'].get('approach', 'general')}
- Steps completed: {state['step_count']}
- Computation time: {state['solution'].get('compute_time', 0):.4f}s
- Platform: {state['solution'].get('method', 'MLX on Apple Silicon')}

✅ Mathematical analysis complete!
"""
        
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": summary}]
        }
    
    # Create the graph
    workflow = StateGraph(MathAgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("computation", computation_node)
    workflow.add_node("summary", summary_node)
    
    # Define edges
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "computation")
    workflow.add_edge("computation", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile()


def format_math_result(result: Dict[str, Any], approach: str) -> str:
    """Format mathematical results for display"""
    
    if approach == "linear_system":
        return f"""
🔢 **Linear System Solution**:
• Equation: 2x + y = 5, x + 3y = 7
• Solution: x = {result['solution'][0]:.3f}, y = {result['solution'][1]:.3f}
• Verification: Ax = [{result['verification'][0]:.3f}, {result['verification'][1]:.3f}]
• Solve time: {result['solve_time']:.4f} seconds
• Method: {result['method']}
"""
    
    elif approach == "matrix_analysis":
        return f"""
📊 **Matrix Analysis**:
• Shape: {result['matrix_shape']}
• Trace: {result['trace']:.3f}
• Frobenius Norm: {result['frobenius_norm']:.3f}  
• Determinant: {result['determinant']:.3f}
• Compute time: {result['compute_time']:.4f} seconds
• Platform: {result['platform']}
"""
    
    elif approach == "statistics":
        return f"""
📈 **Statistical Analysis**:
• Sample size: {result['count']} data points
• Mean: {result['mean']:.3f}
• Std Dev: {result['std_dev']:.3f}
• Range: [{result['min']:.3f}, {result['max']:.3f}]
• Median: {result['median']:.3f}
• Sum: {result['sum']:.3f}
• Compute time: {result['compute_time']:.4f} seconds
• Method: {result['method']}
"""
    
    else:
        return "✅ Computation completed successfully!"


def run_standalone_demo():
    """Run demo without LangGraph (direct MLX computation)"""
    print("🔧 Running in standalone mode (MLX only)")
    print("=" * 50)
    
    processor = MLXMathProcessor()
    
    # Demo 1: Linear System
    print("\n1️⃣ Linear System Demo:")
    result = processor.solve_linear_system("solve system")
    if result["success"]:
        print(format_math_result(result, "linear_system"))
    else:
        print(f"❌ Error: {result['error']}")
    
    # Demo 2: Matrix Analysis
    print("\n2️⃣ Matrix Analysis Demo:")
    result = processor.matrix_analysis("analyze matrix")
    if result["success"]:
        print(format_math_result(result, "matrix_analysis"))
    else:
        print(f"❌ Error: {result['error']}")
    
    # Demo 3: Statistics
    print("\n3️⃣ Statistical Analysis Demo:")
    result = processor.statistical_analysis("analyze data")
    if result["success"]:
        print(format_math_result(result, "statistics"))
    else:
        print(f"❌ Error: {result['error']}")


def run_agent_demo():
    """Run the full LangGraph agent demo"""
    print("🤖 Running LangGraph + MLX Agent Demo")
    print("=" * 50)
    
    agent = create_simple_agent()
    if not agent:
        print("❌ Agent creation failed")
        return
    
    # Demo problems
    problems = [
        "solve linear system equations",
        "analyze matrix properties", 
        "perform statistical analysis"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}️⃣ **Problem**: {problem}")
        print("-" * 30)
        
        # Initialize state
        initial_state = {
            "messages": [{"role": "user", "content": problem}],
            "current_problem": problem,
            "solution": {},
            "step_count": 0
        }
        
        try:
            # Run agent
            result = agent.invoke(initial_state)
            
            # Display conversation
            for message in result["messages"]:
                role = "🧑" if message["role"] == "user" else "🤖"
                print(f"\n{role} {message['content']}")
                
        except Exception as e:
            print(f"❌ Agent error: {e}")
        
        print("\n" + "="*50)


def interactive_demo():
    """Interactive demo mode"""
    print("\n🎮 Interactive Mode")
    print("Commands: 'linear', 'matrix', 'stats', 'quit'")
    
    processor = MLXMathProcessor()
    
    while True:
        user_input = input("\n🗣️  Your request: ").strip().lower()
        
        if user_input in ['quit', 'exit', 'q']:
            break
        
        print(f"\n🔄 Processing '{user_input}'...")
        
        if 'linear' in user_input or 'system' in user_input:
            result = processor.solve_linear_system(user_input)
            if result["success"]:
                print(format_math_result(result, "linear_system"))
        elif 'matrix' in user_input:
            result = processor.matrix_analysis(user_input)
            if result["success"]:
                print(format_math_result(result, "matrix_analysis"))
        elif 'stat' in user_input or 'data' in user_input:
            result = processor.statistical_analysis(user_input)
            if result["success"]:
                print(format_math_result(result, "statistics"))
        else:
            print("🤔 Try: 'linear system', 'matrix analysis', or 'statistics'")


def main():
    """Main demo function"""
    print("""
🚀 LangGraph + MLX Integration Demo
===================================

This demo showcases:
✅ MLX mathematical computation on Apple Silicon
✅ LangGraph agent workflows (if available)
✅ Local processing with zero external dependencies
✅ Interactive mathematical problem solving

Available demos:
1. Standalone MLX computation
2. LangGraph agent workflow (if available)  
3. Interactive mode
""")
    
    if LANGGRAPH_AVAILABLE:
        print("🎯 LangGraph detected - full agent demo available!")
        
        while True:
            choice = input("\nChoose demo (1/2/3/q): ").strip()
            
            if choice == '1':
                run_standalone_demo()
            elif choice == '2':
                run_agent_demo()
            elif choice == '3':
                interactive_demo()
            elif choice.lower() in ['q', 'quit']:
                break
            else:
                print("Please choose 1, 2, 3, or q")
    else:
        print("🔧 Running standalone MLX demo (install LangGraph for full features)")
        run_standalone_demo()
        interactive_demo()
    
    print("\n👋 Thanks for trying the LangGraph + MLX demo!")


if __name__ == "__main__":
    main()
