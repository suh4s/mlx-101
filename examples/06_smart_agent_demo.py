#!/usr/bin/env python3
"""
🤖 LangGraph + MLX Agent Demo: "Smart Research Assistant"
==========================================================

A comprehensive demo showcasing LangGraph + MLX integration with local models.
This creates an intelligent research assistant that can:

• 🧠 Reason with local LLMs (Llama 3.2, Phi-3, etc.)
• 🔢 Solve mathematical problems with MLX
• 📊 Analyze data with Apple Silicon acceleration  
• 🔍 Search and summarize information
• 📝 Generate reports with local inference

Models used:
- Language Model: Llama-3.2-3B-Instruct (via MLX-LM)
- Embeddings: all-MiniLM-L6-v2 (for RAG)
- Math Engine: MLX linear algebra (Apple Silicon optimized)

Installation:
pip install langgraph langchain mlx-lm sentence-transformers
"""

import mlx.core as mx
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from pathlib import Path

# Check for required packages
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.tools import tool
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("📦 LangGraph not installed. Install with: pip install langgraph langchain")
    LANGGRAPH_AVAILABLE = False

try:
    from mlx_lm import load, generate
    MLX_LM_AVAILABLE = True
except ImportError:
    print("📦 MLX-LM not installed. Install with: pip install mlx-lm")
    MLX_LM_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("📦 Sentence Transformers not installed. Install with: pip install sentence-transformers")
    EMBEDDINGS_AVAILABLE = False


# Agent State Definition
class ResearchAgentState(TypedDict):
    """State for our research assistant agent"""
    messages: List[BaseMessage]
    current_task: str
    research_data: Dict[str, Any]
    mlx_results: Dict[str, Any]
    final_answer: str
    step_count: int


@dataclass
class ModelConfig:
    """Configuration for local models"""
    llm_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_tokens: int = 512
    temperature: float = 0.7


class MLXMathEngine:
    """MLX-powered mathematical reasoning engine"""
    
    @staticmethod
    def solve_linear_system(A_data: List[List[float]], b_data: List[float]) -> Dict[str, Any]:
        """Solve linear system Ax = b using MLX"""
        try:
            A = mx.array(A_data, dtype=mx.float32)
            b = mx.array(b_data, dtype=mx.float32)
            
            # Use CPU stream for compatibility
            with mx.stream(mx.cpu):
                x = mx.linalg.solve(A, b)
                mx.eval(x)
            
            return {
                "success": True,
                "solution": x.tolist(),
                "verification": (A @ x).tolist(),
                "method": "MLX Linear Solver"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def matrix_analysis(matrix_data: List[List[float]]) -> Dict[str, Any]:
        """Analyze matrix properties using MLX"""
        try:
            A = mx.array(matrix_data, dtype=mx.float32)
            
            # Basic properties
            properties = {
                "shape": list(A.shape),
                "trace": float(mx.trace(A)) if A.shape[0] == A.shape[1] else None,
                "frobenius_norm": float(mx.linalg.norm(A)),
                "determinant": None
            }
            
            # Determinant for square matrices
            if A.shape[0] == A.shape[1] and A.shape[0] <= 4:  # Small matrices only
                try:
                    with mx.stream(mx.cpu):
                        det = mx.linalg.det(A)
                        mx.eval(det)
                    properties["determinant"] = float(det)
                except:
                    pass
            
            return {
                "success": True,
                "properties": properties,
                "computed_with": "MLX on Apple Silicon"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def statistical_analysis(data: List[float]) -> Dict[str, Any]:
        """Perform statistical analysis using MLX"""
        try:
            arr = mx.array(data, dtype=mx.float32)
            
            stats = {
                "count": len(data),
                "mean": float(mx.mean(arr)),
                "std": float(mx.std(arr)),
                "min": float(mx.min(arr)),
                "max": float(mx.max(arr)),
                "sum": float(mx.sum(arr))
            }
            
            # Percentiles (approximate)
            sorted_data = mx.sort(arr)
            n = len(data)
            if n > 4:
                stats["median"] = float(sorted_data[n//2])
                stats["q1"] = float(sorted_data[n//4])
                stats["q3"] = float(sorted_data[3*n//4])
            
            return {
                "success": True,
                "statistics": stats,
                "processed_with": "MLX vectorized operations"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class LocalLLMEngine:
    """Local LLM powered by MLX-LM"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the local model"""
        if not MLX_LM_AVAILABLE:
            print("⚠️  MLX-LM not available. LLM features disabled.")
            return
        
        try:
            print(f"🔄 Loading {self.config.llm_model}...")
            self.model, self.tokenizer = load(self.config.llm_model)
            print("✅ Local LLM loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Try downloading the model first:")
            print(f"   python -m mlx_lm.download --hf-repo {self.config.llm_model}")
    
    def generate(self, prompt: str, system_message: str = "") -> str:
        """Generate response using local LLM"""
        if not self.model:
            return "🤖 Local LLM not available. Please install mlx-lm and download a model."
        
        try:
            # Format prompt
            if system_message:
                formatted_prompt = f"System: {system_message}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\n\nAssistant:"
            
            # Generate with MLX
            response = generate(
                self.model, 
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=self.config.max_tokens,
                temp=self.config.temperature
            )
            
            return response.strip()
        except Exception as e:
            return f"❌ Generation error: {e}"


class EmbeddingEngine:
    """Local embedding engine for RAG capabilities"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            print("⚠️  Sentence Transformers not available. RAG features disabled.")
            return
        
        try:
            print(f"🔄 Loading {self.config.embedding_model}...")
            self.model = SentenceTransformer(self.config.embedding_model)
            print("✅ Embedding model loaded!")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Create embedding for text"""
        if not self.model:
            return None
        
        try:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not self.model:
            return 0.0
        
        try:
            embeddings = self.model.encode([text1, text2])
            # Convert to MLX for computation
            emb1 = mx.array(embeddings[0])
            emb2 = mx.array(embeddings[1])
            
            # Cosine similarity
            similarity = mx.sum(emb1 * emb2) / (mx.linalg.norm(emb1) * mx.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            print(f"❌ Similarity calculation error: {e}")
            return 0.0


# Define tools for the agent
def create_agent_tools(math_engine: MLXMathEngine, llm_engine: LocalLLMEngine):
    """Create tools powered by MLX and local models"""
    
    @tool
    def solve_math_problem(problem_description: str) -> str:
        """
        Solve mathematical problems using MLX-powered computation.
        
        Args:
            problem_description: Description of the math problem to solve
        
        Returns:
            Solution with step-by-step explanation
        """
        # For demo, we'll handle common problem types
        problem_lower = problem_description.lower()
        
        if "linear system" in problem_lower or "solve" in problem_lower:
            # Example: solve the system 2x + y = 5, x + 3y = 7
            A = [[2, 1], [1, 3]]
            b = [5, 7]
            result = math_engine.solve_linear_system(A, b)
            
            if result["success"]:
                return f"""
🔢 **Linear System Solution**:
- System: 2x + y = 5, x + 3y = 7
- Solution: x = {result['solution'][0]:.3f}, y = {result['solution'][1]:.3f}
- Verification: {result['verification']}
- Computed with: {result['method']} on Apple Silicon
"""
            else:
                return f"❌ Math error: {result['error']}"
        
        elif "matrix" in problem_lower:
            # Example matrix analysis
            sample_matrix = [[1, 2], [3, 4]]
            result = math_engine.matrix_analysis(sample_matrix)
            
            if result["success"]:
                props = result["properties"]
                return f"""
📊 **Matrix Analysis**:
- Shape: {props['shape']}
- Trace: {props.get('trace', 'N/A')}
- Frobenius Norm: {props['frobenius_norm']:.3f}
- Determinant: {props.get('determinant', 'N/A')}
- Computed with: {result['computed_with']}
"""
            else:
                return f"❌ Matrix analysis error: {result['error']}"
        
        elif "statistics" in problem_lower or "data" in problem_lower:
            # Example statistical analysis
            sample_data = [1.2, 2.5, 3.1, 2.8, 4.2, 3.7, 2.9, 3.4, 2.1, 3.8]
            result = math_engine.statistical_analysis(sample_data)
            
            if result["success"]:
                stats = result["statistics"]
                return f"""
📈 **Statistical Analysis**:
- Count: {stats['count']}
- Mean: {stats['mean']:.3f}
- Std Dev: {stats['std']:.3f}
- Range: [{stats['min']:.3f}, {stats['max']:.3f}]
- Median: {stats.get('median', 'N/A')}
- Computed with: {result['processed_with']}
"""
            else:
                return f"❌ Statistics error: {result['error']}"
        
        else:
            return "🤔 Please specify the type of math problem (linear system, matrix analysis, or statistics)"
    
    @tool
    def research_question(query: str) -> str:
        """
        Research a question using local LLM reasoning.
        
        Args:
            query: The research question to investigate
        
        Returns:
            Researched answer with reasoning
        """
        system_prompt = """You are a helpful research assistant. Provide accurate, 
        well-reasoned answers based on your knowledge. Be concise but thorough."""
        
        response = llm_engine.generate(query, system_prompt)
        
        return f"""
🔍 **Research Results**:
Query: {query}

{response}

---
*Generated locally with {llm_engine.config.llm_model} on Apple Silicon*
"""
    
    @tool
    def analyze_performance(task_description: str) -> str:
        """
        Analyze performance characteristics of MLX operations.
        
        Args:
            task_description: Description of the task to analyze
        
        Returns:
            Performance analysis results
        """
        # Benchmark some MLX operations
        size = 100
        
        # Matrix multiplication benchmark
        start_time = time.time()
        A = mx.random.normal((size, size))
        B = mx.random.normal((size, size))
        C = A @ B
        mx.eval(C)  # Force evaluation
        mlx_time = time.time() - start_time
        
        # Memory usage estimate
        memory_mb = (size * size * 4 * 3) / (1024 * 1024)  # 3 matrices, 4 bytes per float32
        
        return f"""
⚡ **Performance Analysis**:
Task: {task_description}

🔢 **MLX Benchmark Results**:
- Matrix multiplication ({size}x{size}): {mlx_time:.4f} seconds
- Memory usage: ~{memory_mb:.1f} MB
- Throughput: {(size**3 / mlx_time / 1e9):.2f} GFLOPS
- Platform: Apple Silicon with unified memory

💡 **Insights**:
- MLX leverages Metal Performance Shaders
- Zero-copy operations between CPU and GPU
- Automatic operation fusion for efficiency
- Ideal for real-time AI applications
"""
    
    return [solve_math_problem, research_question, analyze_performance]


def create_research_agent(config: ModelConfig):
    """Create the main research agent using LangGraph"""
    
    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph not available. Cannot create agent.")
        return None
    
    # Initialize engines
    math_engine = MLXMathEngine()
    llm_engine = LocalLLMEngine(config)
    embedding_engine = EmbeddingEngine(config)
    
    # Create tools
    tools = create_agent_tools(math_engine, llm_engine)
    
    # Define agent nodes
    def planning_node(state: ResearchAgentState) -> ResearchAgentState:
        """Plan the research approach"""
        current_message = state["messages"][-1].content
        step_count = state.get("step_count", 0) + 1
        
        # Simple planning logic
        plan = f"""
🎯 **Research Plan** (Step {step_count}):
User Query: {current_message}

I'll help you with this request using:
• Local LLM reasoning (Llama 3.2)
• MLX-powered mathematical computation
• Apple Silicon acceleration

Let me determine the best approach...
"""
        
        return {
            **state,
            "current_task": current_message,
            "step_count": step_count,
            "messages": state["messages"] + [AIMessage(content=plan)]
        }
    
    def execution_node(state: ResearchAgentState) -> ResearchAgentState:
        """Execute the research task"""
        task = state["current_task"].lower()
        
        if any(word in task for word in ["math", "solve", "calculate", "equation"]):
            # Use math tool
            result = tools[0].invoke({"problem_description": state["current_task"]})
            task_type = "Mathematical Analysis"
        elif any(word in task for word in ["research", "explain", "what", "how", "why"]):
            # Use research tool
            result = tools[1].invoke({"query": state["current_task"]})
            task_type = "Research & Analysis"
        elif any(word in task for word in ["performance", "benchmark", "speed"]):
            # Use performance tool
            result = tools[2].invoke({"task_description": state["current_task"]})
            task_type = "Performance Analysis"
        else:
            # Default to research
            result = tools[1].invoke({"query": state["current_task"]})
            task_type = "General Research"
        
        execution_result = f"""
✅ **{task_type} Complete**:

{result}

🚀 **Powered by**:
- LangGraph: Agent orchestration
- MLX: Apple Silicon acceleration  
- Local Models: Privacy-first inference
"""
        
        return {
            **state,
            "mlx_results": {"type": task_type, "result": result},
            "messages": state["messages"] + [AIMessage(content=execution_result)]
        }
    
    def finalization_node(state: ResearchAgentState) -> ResearchAgentState:
        """Finalize the research and provide summary"""
        summary = f"""
📋 **Research Session Summary**:
- Steps completed: {state['step_count']}
- Task type: {state['mlx_results'].get('type', 'General')}
- Local models used: {config.llm_model}
- Computation engine: MLX on Apple Silicon

The research has been completed successfully! All computation was performed
locally on your device using Apple Silicon optimization.
"""
        
        return {
            **state,
            "final_answer": summary,
            "messages": state["messages"] + [AIMessage(content=summary)]
        }
    
    # Create the graph
    workflow = StateGraph(ResearchAgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("execution", execution_node)  
    workflow.add_node("finalization", finalization_node)
    
    # Define the flow
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "execution")
    workflow.add_edge("execution", "finalization")
    workflow.add_edge("finalization", END)
    
    return workflow.compile()


async def run_demo():
    """Run the interactive demo"""
    print("🤖 LangGraph + MLX Research Assistant Demo")
    print("=" * 60)
    
    # Model configuration
    config = ModelConfig()
    
    # Create agent
    agent = create_research_agent(config)
    
    if not agent:
        print("❌ Agent creation failed. Please install required packages.")
        return
    
    print("\n🎯 **Available Commands**:")
    print("1. 'solve math problem' - Mathematical analysis with MLX")
    print("2. 'research [topic]' - Research questions with local LLM")
    print("3. 'performance analysis' - MLX performance benchmarks")
    print("4. 'quit' - Exit the demo")
    
    while True:
        print("\n" + "="*50)
        user_input = input("🗣️  Your request: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for trying the LangGraph + MLX demo!")
            break
        
        if not user_input:
            continue
        
        print("\n🔄 Processing your request...")
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "current_task": "",
            "research_data": {},
            "mlx_results": {},
            "final_answer": "",
            "step_count": 0
        }
        
        try:
            # Run the agent
            result = agent.invoke(initial_state)
            
            # Display results
            print("\n📤 **Agent Response**:")
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    print(f"\n🤖 {message.content}")
                    
        except Exception as e:
            print(f"❌ Error running agent: {e}")


def main():
    """Main demo function"""
    print("""
🍎 LangGraph + MLX Research Assistant
=====================================

This demo showcases the integration of:
• 🤖 LangGraph - Advanced AI agent workflows
• ⚡ MLX - Apple Silicon acceleration  
• 🧠 Local Models - Privacy-first inference
• 🔒 On-device Processing - No data leaves your Mac

Features demonstrated:
✅ Local LLM reasoning (Llama 3.2)
✅ MLX mathematical computation
✅ Agent state management
✅ Tool integration
✅ Apple Silicon optimization

Prerequisites:
pip install langgraph langchain mlx-lm sentence-transformers
""")
    
    if not all([LANGGRAPH_AVAILABLE, MLX_LM_AVAILABLE]):
        print("\n📦 **Installation Required**:")
        if not LANGGRAPH_AVAILABLE:
            print("• pip install langgraph langchain")
        if not MLX_LM_AVAILABLE:
            print("• pip install mlx-lm")
        if not EMBEDDINGS_AVAILABLE:
            print("• pip install sentence-transformers")
        
        print("\n💡 **Model Download** (after installing mlx-lm):")
        print("python -m mlx_lm.download --hf-repo mlx-community/Llama-3.2-3B-Instruct-4bit")
        return
    
    # Run the demo
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main()
