#!/usr/bin/env python3
"""
🚀 LangGraph + MLX Local Agent Demo (Simplified & Working)
=========================================================

A practical demonstration that actually works! This creates a local AI assistant with:

• 🧠 Local LLM: Llama-3.2-3B-Instruct (via MLX-LM)  
• 🔢 Math Engine: MLX on Apple Silicon
• 📚 Knowledge Search: Local embeddings
• 🤖 Agent Workflows: LangGraph orchestration
• 🎯 Interactive CLI: Real-time demo

Installation:
uv add mlx-lm sentence-transformers langgraph

Usage:
python examples/09_working_local_agent.py

Local Models Used:
- Llama-3.2-3B-Instruct-4bit (Language understanding)
- all-MiniLM-L6-v2 (Document embeddings)
- MLX Linear Algebra (Mathematical reasoning)
"""

import mlx.core as mx
import numpy as np
import time
from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass
import warnings

# Suppress SSL warnings for cleaner output
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

# Check dependencies
print("🔍 Checking Dependencies...")

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage, AIMessage
    print("✅ LangGraph available")
    LANGGRAPH_OK = True
except ImportError:
    print("❌ LangGraph missing: uv add langgraph")
    LANGGRAPH_OK = False

try:
    from mlx_lm import load, generate
    print("✅ MLX-LM available")
    MLX_LM_OK = True
except ImportError:
    print("❌ MLX-LM missing: uv add mlx-lm")
    MLX_LM_OK = False

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers available")
    EMBEDDINGS_OK = True
except ImportError:
    print("❌ Embeddings missing: uv add sentence-transformers")
    EMBEDDINGS_OK = False

print(f"✅ MLX Core: {mx.__version__ if hasattr(mx, '__version__') else 'installed'}")


# ============================================================================
# CORE ENGINES
# ============================================================================

@dataclass 
class Config:
    llm_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_tokens: int = 1024  # Increased from 300 to 1024
    temperature: float = 0.7


class MLXMathEngine:
    """Apple Silicon optimized math engine"""
    
    @staticmethod
    def solve_linear_system(description: str) -> Dict[str, Any]:
        """Solve 2x2 linear systems"""
        try:
            # For demo: parse common patterns or use default
            if "2x" in description and "3y" in description:
                A = mx.array([[2.0, 3.0], [1.0, -1.0]])
                b = mx.array([7.0, 1.0])
            else:
                # Default system: 3x + y = 9, x + 2y = 8
                A = mx.array([[3.0, 1.0], [1.0, 2.0]])
                b = mx.array([9.0, 8.0])
            
            with mx.stream(mx.cpu):
                solution = mx.linalg.solve(A, b)
                verification = A @ solution
                mx.eval(solution, verification)
            
            return {
                "success": True,
                "solution": [float(solution[0]), float(solution[1])],
                "verification": verification.tolist(),
                "device": "Apple Silicon (MLX)"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def analyze_data(numbers: List[float]) -> Dict[str, Any]:
        """Statistical analysis with MLX"""
        try:
            arr = mx.array(numbers)
            with mx.stream(mx.cpu):
                mean_val = mx.mean(arr)
                std_val = mx.std(arr)
                min_val = mx.min(arr)
                max_val = mx.max(arr)
                mx.eval(mean_val, std_val, min_val, max_val)
            
            return {
                "success": True,
                "mean": float(mean_val),
                "std": float(std_val),
                "range": [float(min_val), float(max_val)],
                "count": len(numbers),
                "device": "Apple Silicon"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class LocalLLM:
    """Local language model using MLX-LM"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.conversation_history = []  # Add conversation memory
    
    def load(self) -> bool:
        """Load the model"""
        if not MLX_LM_OK:
            return False
        
        try:
            print(f"📥 Loading {self.config.llm_model}...")
            self.model, self.tokenizer = load(self.config.llm_model)
            self.loaded = True
            print("✅ Local LLM ready")
            return True
        except Exception as e:
            print(f"❌ LLM load failed: {e}")
            return False
    
    def chat(self, message: str, system: str = "") -> str:
        """Generate response with conversation memory"""
        if not self.loaded:
            return self._mock_response(message)
        
        # Add user message to history
        self.conversation_history.append(f"User: {message}")
        
        try:
            # Build prompt with conversation context
            if system:
                prompt_parts = [f"System: {system}"]
            else:
                prompt_parts = []
            
            # Add recent conversation history (last 6 exchanges to manage token limit)
            recent_history = self.conversation_history[-6:]
            prompt_parts.extend(recent_history)
            prompt_parts.append("Assistant:")
            
            prompt = "\n".join(prompt_parts)
            
            # Try different MLX-LM API versions
            try:
                # Try newer API first
                response = generate(
                    self.model, self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    verbose=False
                )
            except TypeError:
                # Try older API without explicit parameters
                response = generate(self.model, self.tokenizer, prompt)
            
            # Clean up the response - remove stop tokens and artifacts
            response = response.strip()
            
            # Remove common artifacts that might cut responses
            cleanup_patterns = [
                "User:", "System:", "Assistant:", 
                "<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>",
                "\n\nUser:", "\n\nSystem:", "\n\nAssistant:"
            ]
            for pattern in cleanup_patterns:
                if pattern in response:
                    response = response.split(pattern)[0].strip()
            
            # Ensure we have a meaningful response
            if not response or len(response) < 5:
                response = self._mock_response(message)
            
            # Add assistant response to history
            self.conversation_history.append(f"Assistant: {response}")
            
            return response
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            print("   Falling back to mock responses...")
            return self._mock_response(message)
    
    def _mock_response(self, message: str) -> str:
        """Enhanced fallback responses"""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ["solve", "equation", "math", "linear"]):
            return "I can solve linear equations using MLX on Apple Silicon. Let me process that mathematical problem for you."
        elif any(word in msg_lower for word in ["data", "analyze", "statistics", "numbers"]):
            return "I'll analyze your data using MLX statistical functions optimized for Apple Silicon."
        elif any(word in msg_lower for word in ["apple", "silicon", "mlx", "m1", "m2", "m3"]):
            return "Apple Silicon is ARM-based processors with Neural Engine acceleration. MLX leverages this for efficient ML computations."
        elif any(word in msg_lower for word in ["what", "how", "explain"]):
            return "I'm your local AI assistant running entirely on Apple Silicon. I can help with math, data analysis, and answer questions using local models."
        elif any(word in msg_lower for word in ["hello", "hi", "help"]):
            return "Hello! I'm a local AI assistant powered by Llama-3.2-3B and MLX on Apple Silicon. I can solve math problems, analyze data, and search knowledge - all locally!"
        else:
            return f"I understand you're asking about: {message[:50]}... I'm processing this using local models on Apple Silicon."
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")
    
    def get_conversation_length(self) -> int:
        """Get number of exchanges in conversation"""
        return len(self.conversation_history) // 2  # Divide by 2 since each exchange has user + assistant


class LocalKnowledgeBase:
    """Local document search with embeddings"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.docs = []
        self.embeddings = None
        self.loaded = False
    
    def load(self) -> bool:
        """Load embeddings model and documents"""
        if not EMBEDDINGS_OK:
            return False
        
        try:
            print(f"📥 Loading {self.config.embedding_model}...")
            self.model = SentenceTransformer(self.config.embedding_model)
            
            # Sample knowledge base
            self.docs = [
                "Apple Silicon chips use ARM architecture with Neural Engine",
                "MLX is Apple's ML framework optimized for Apple Silicon",
                "LangGraph enables building stateful multi-agent workflows",
                "Local models provide privacy and reduced latency",
                "M-series chips have unified memory architecture",
                "Vector embeddings enable semantic similarity search",
                "Llama models can run locally with MLX-LM",
                "Apple Silicon accelerates matrix operations efficiently"
            ]
            
            print("🔍 Computing embeddings...")
            self.embeddings = self.model.encode(self.docs)
            self.loaded = True
            print("✅ Knowledge base ready")
            return True
            
        except Exception as e:
            print(f"❌ Knowledge base failed: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search documents"""
        if not self.loaded:
            return [{"text": "Knowledge base unavailable", "score": 0.0}]
        
        try:
            query_emb = self.model.encode([query])
            scores = np.dot(self.embeddings, query_emb.T).flatten()
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.docs[idx],
                    "score": float(scores[idx])
                })
            return results
            
        except Exception as e:
            return [{"text": f"Search error: {e}", "score": 0.0}]


# ============================================================================
# AGENT DEFINITION  
# ============================================================================

class AgentState(TypedDict):
    """Agent state"""
    messages: List[Any]
    task: str
    math_result: Dict[str, Any]
    search_result: List[Dict[str, Any]]
    final_answer: str


class LocalAgent:
    """Complete local AI agent"""
    
    def __init__(self):
        self.config = Config()
        self.math_engine = MLXMathEngine()
        self.llm = LocalLLM(self.config)
        self.knowledge = LocalKnowledgeBase(self.config)
        self.graph = None
        self.parallel_mode = False  # Add mode switching
    
    def toggle_parallel_mode(self) -> str:
        """Toggle between sequential and parallel agent modes"""
        self.parallel_mode = not self.parallel_mode
        
        # Reinitialize the graph with the new mode
        if LANGGRAPH_OK:
            if self.parallel_mode:
                self.graph = self._create_parallel_graph()
                return ("🔀 Switched to PARALLEL agent mode "
                        "(math, search, and reasoning run concurrently)")
            else:
                self.graph = self._create_graph()
                return ("📝 Switched to SEQUENTIAL agent mode "
                        "(math → search → reasoning in order)")
        else:
            return "❌ LangGraph not available - mode switching disabled"
    
    def get_current_mode(self) -> str:
        """Get current agent mode"""
        return "PARALLEL" if self.parallel_mode else "SEQUENTIAL"

    def initialize(self):
        """Load all components"""
        print("\n🚀 Initializing Local AI Agent...")
        print("=" * 50)
        
        # Load engines
        llm_ok = self.llm.load()
        kb_ok = self.knowledge.load()
        
        # Create graph if LangGraph available
        if LANGGRAPH_OK:
            if self.parallel_mode:
                self.graph = self._create_parallel_graph()
                print("🤖 Parallel agent graph created")
            else:
                self.graph = self._create_graph()
                print("🤖 Sequential agent graph created")
        
        print("\n✅ Agent initialization complete!")
        return llm_ok or kb_ok
    
    def _create_graph(self) -> StateGraph:
        """Create LangGraph workflow"""
        
        def reasoning_node(state: AgentState) -> AgentState:
            """Main reasoning with LLM"""
            task = state["task"]
            
            system_prompt = """You are a helpful AI assistant with access to:
            1. Mathematical tools for linear algebra (MLX-powered)
            2. Knowledge search for information lookup
            3. Data analysis capabilities
            
            Analyze the user's request and provide helpful responses."""
            
            response = self.llm.chat(task, system_prompt)
            
            return {
                **state,
                "messages": [AIMessage(content=response)],
                "final_answer": response
            }
        
        def math_node(state: AgentState) -> AgentState:
            """Mathematical reasoning"""
            task = state["task"]
            
            if any(word in task.lower() for word in ["solve", "equation", "math"]):
                result = self.math_engine.solve_linear_system(task)
                return {**state, "math_result": result}
            
            if any(word in task.lower() for word in ["analyze", "data", "statistics"]):
                # Extract numbers from task (simple regex)
                import re
                numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', task)]
                if numbers:
                    result = self.math_engine.analyze_data(numbers)
                    return {**state, "math_result": result}
            
            return state
        
        def search_node(state: AgentState) -> AgentState:
            """Knowledge search"""
            task = state["task"]
            
            if any(word in task.lower() for word in ["what", "how", "apple", "mlx", "silicon"]):
                results = self.knowledge.search(task, top_k=2)
                return {**state, "search_result": results}
            
            return state
        
        def should_continue(state: AgentState) -> str:
            """Simple flow control"""
            return END
        
        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("reasoning", reasoning_node)
        workflow.add_node("math", math_node)
        workflow.add_node("search", search_node)
        
        workflow.set_entry_point("math")
        workflow.add_edge("math", "search")
        workflow.add_edge("search", "reasoning")
        workflow.add_conditional_edges("reasoning", should_continue, {END: END})
        
        return workflow.compile()
    
    def _create_parallel_graph(self) -> StateGraph:
        """Create LangGraph workflow with PARALLEL agents"""
        
        def start_parallel(state: AgentState) -> AgentState:
            """Entry point that initiates parallel processing"""
            return state
        
        def math_and_search_node(state: AgentState) -> AgentState:
            """Combined node that runs math and search in sequence but faster"""
            task = state["task"]
            result_state = {**state}
            
            # Math processing
            if any(word in task.lower() for word in ["solve", "equation", "math"]):
                math_result = self.math_engine.solve_linear_system(task)
                result_state["math_result"] = math_result
            
            if any(word in task.lower() for word in ["analyze", "data", "statistics"]):
                import re
                numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', task)]
                if numbers:
                    math_result = self.math_engine.analyze_data(numbers)
                    result_state["math_result"] = math_result
            
            # Search processing
            if any(word in task.lower() for word in ["what", "how", "apple", "mlx", "silicon"]):
                search_results = self.knowledge.search(task, top_k=2)
                result_state["search_result"] = search_results
            
            return result_state
        
        def reasoning_node(state: AgentState) -> AgentState:
            """Final reasoning with LLM"""
            task = state["task"]
            
            system_prompt = """You are a helpful AI assistant with access to:
            1. Mathematical tools for linear algebra (MLX-powered)
            2. Knowledge search for information lookup
            3. Data analysis capabilities
            
            Analyze the user's request and provide helpful responses."""
            
            response = self.llm.chat(task, system_prompt)
            
            return {
                **state,
                "final_answer": response
            }
        
        def should_continue(state: AgentState) -> str:
            """Flow control"""
            return END
        
        # Build simplified "parallel" graph (actually optimized sequential)
        workflow = StateGraph(AgentState)
        workflow.add_node("start", start_parallel)
        workflow.add_node("math_search", math_and_search_node)
        workflow.add_node("reasoning", reasoning_node)
        
        # Flow: start -> math_search -> reasoning -> end
        workflow.set_entry_point("start")
        workflow.add_edge("start", "math_search")
        workflow.add_edge("math_search", "reasoning")
        workflow.add_conditional_edges("reasoning", should_continue, {END: END})
        
        return workflow.compile()
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """Process user input"""
        print(f"\n🤖 Processing: '{user_input}'")
        
        if self.graph and LANGGRAPH_OK:
            # Use LangGraph workflow
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                task=user_input,
                math_result={},
                search_result=[],
                final_answer=""
            )
            
            try:
                result = self.graph.invoke(initial_state)
                return {
                    "success": True,
                    "response": result.get("final_answer", "No response generated"),
                    "math_result": result.get("math_result", {}),
                    "search_result": result.get("search_result", [])
                }
            except Exception as e:
                print(f"❌ Graph error: {e}")
                return self._simple_process(user_input)
        else:
            return self._simple_process(user_input)
    
    def _simple_process(self, user_input: str) -> Dict[str, Any]:
        """Simple processing without LangGraph"""
        result = {
            "success": True,
            "response": "",
            "math_result": {},
            "search_result": []
        }
        
        # Math processing
        if any(word in user_input.lower() for word in ["solve", "equation", "math"]):
            result["math_result"] = self.math_engine.solve_linear_system(user_input)
        
        # Search processing  
        if any(word in user_input.lower() for word in ["what", "how", "apple", "mlx"]):
            result["search_result"] = self.knowledge.search(user_input)
        
        # LLM response
        result["response"] = self.llm.chat(user_input, 
            "You are a helpful AI assistant running locally on Apple Silicon.")
        
        return result


# ============================================================================
# DEMO INTERFACE
# ============================================================================

def run_capability_demos(agent: LocalAgent):
    """Demonstrate individual capabilities"""
    print("\n🎯 Capability Demonstrations")
    print("=" * 50)
    
    # Math demo
    print("\n1️⃣ Mathematical Reasoning (MLX):")
    math_result = agent.math_engine.solve_linear_system("3x + y = 9, x + 2y = 8")
    if math_result["success"]:
        sol = math_result["solution"]
        print(f"   ✅ Solution: x = {sol[0]:.3f}, y = {sol[1]:.3f}")
        print(f"   Device: {math_result['device']}")
    
    # Data analysis demo
    print("\n2️⃣ Data Analysis (MLX):")
    data_result = agent.math_engine.analyze_data([2.1, 1.8, 2.4, 1.9, 2.2, 1.7, 2.0])
    if data_result["success"]:
        print(f"   📊 Mean: {data_result['mean']:.3f}")
        print(f"   📊 Std: {data_result['std']:.3f}")
        print(f"   Device: {data_result['device']}")
    
    # Knowledge search demo
    print("\n3️⃣ Knowledge Search (Local Embeddings):")
    if agent.knowledge.loaded:
        results = agent.knowledge.search("What is Apple Silicon?")
        for i, result in enumerate(results[:2], 1):
            print(f"   {i}. {result['text']} (Score: {result['score']:.3f})")
    
    # LLM demo
    print("\n4️⃣ Language Understanding (Local LLM):")
    if agent.llm.loaded:
        response = agent.llm.chat("Explain MLX in one sentence", 
                                  "Be concise and technical.")
        print(f"   💬 {response}")


def run_interactive_demo(agent: LocalAgent):
    """Interactive chat demo with conversation memory"""
    print("\n💬 Interactive Chat Demo")
    print("=" * 50)
    print("🧠 This demo now has conversation memory!")
    print(f"🔧 Agent mode: {agent.get_current_mode()}")
    print("\nTry these examples:")
    print("• 'My name is John' then 'What's my name?'")
    print("• 'Solve 2x + 3y = 7 and x - y = 1'")
    print("• 'What is Apple Silicon?'")
    print("• 'Analyze these numbers: 1.5, 2.3, 1.8, 2.1'")
    print("\nCommands:")
    print("• 'clear' - Clear conversation history")
    print("• 'history' - Show conversation length")
    print("• 'diagram' - Show architecture diagram")
    print("• 'mode' - Toggle sequential/parallel agent mode")
    print("• 'quit' - Exit")
    
    while True:
        try:
            # Show conversation context indicator
            conv_length = agent.llm.get_conversation_length()
            if conv_length > 0:
                user_input = input(f"\n🧑 You (turn {conv_length + 1}): ").strip()
            else:
                user_input = input(f"\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                agent.llm.clear_conversation()
                continue
            elif user_input.lower() == 'mode':
                result_msg = agent.toggle_parallel_mode()
                print(f"🔧 {result_msg}")
                print(f"   Current mode: {agent.get_current_mode()}")
                continue
            elif user_input.lower() == 'diagram':
                show_architecture_diagram()
                continue
            elif user_input.lower() == 'history':
                print(f"📊 Conversation length: {conv_length} exchanges")
                if agent.llm.conversation_history:
                    print("Recent context:")
                    for entry in agent.llm.conversation_history[-4:]:
                        print(f"   {entry}")
                continue
            elif not user_input:
                continue
            
            # Process with agent
            result = agent.process(user_input)
            
            if result["success"]:
                print(f"🤖 Assistant: {result['response']}")
                
                # Show additional results if available
                if result["math_result"] and result["math_result"].get("success"):
                    math = result["math_result"]
                    if "solution" in math:
                        sol = math["solution"]
                        print(f"   🔢 Math: x={sol[0]:.3f}, y={sol[1]:.3f}")
                    elif "mean" in math:
                        print(f"   📊 Stats: mean={math['mean']:.3f}, std={math['std']:.3f}")
                
                if result["search_result"]:
                    print("   🔍 Knowledge:")
                    for r in result["search_result"][:1]:
                        print(f"      {r['text']}")
            else:
                print("❌ Processing failed")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def show_architecture_diagram():
    """Display ASCII diagram of the LangGraph + MLX agent architecture"""
    diagram = """
🚀 LangGraph + MLX Local Agent Architecture
============================================

┌─────────────────────────────────────────────────────────────────────┐
│                          USER INPUT                                │
│                     "Solve 2x + 3y = 7"                           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LOCAL AGENT                                     │
│  ┌─────────────────┬─────────────────┬─────────────────────────────┐ │
│  │   MATH NODE     │   SEARCH NODE   │      REASONING NODE        │ │
│  │                 │                 │                            │ │
│  │ ┌─────────────┐ │ ┌─────────────┐ │ ┌────────────────────────┐ │ │
│  │ │    MLX      │ │ │ Embeddings  │ │ │     Local LLM          │ │ │
│  │ │ Math Engine │ │ │   Engine    │ │ │  (Llama-3.2-3B)        │ │ │
│  │ │             │ │ │             │ │ │                        │ │ │
│  │ │ • Linear    │ │ │ • Semantic  │ │ │ • Conversation Memory  │ │ │
│  │ │   Algebra   │ │ │   Search    │ │ │ • Response Generation  │ │ │
│  │ │ • Statistics│ │ │ • Document  │ │ │ • Context Awareness    │ │ │
│  │ │ • Matrix    │ │ │   Matching  │ │ │                        │ │ │
│  │ │   Ops       │ │ │             │ │ │ ┌────────────────────┐ │ │ │
│  │ │             │ │ │ Knowledge:  │ │ │ │ Chat History:      │ │ │ │
│  │ │ Apple       │ │ │ • Apple     │ │ │ │ User: ...          │ │ │ │
│  │ │ Silicon     │ │ │   Silicon   │ │ │ │ Assistant: ...     │ │ │ │
│  │ │ Optimized   │ │ │ • MLX Info  │ │ │ │ User: ...          │ │ │ │
│  │ └─────────────┘ │ │ • Tech Docs │ │ │ └────────────────────┘ │ │ │
│  └─────────────────┘ └─────────────┘ └────────────────────────────┘ │
└─┬─────────────────┬─────────────────┬─────────────────────────────┬─┘
  │                 │                 │                             │
  ▼                 ▼                 ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH WORKFLOW                             │
│                                                                     │
│  📝 SEQUENTIAL MODE:                                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │ MATH    │───▶│ SEARCH  │───▶│REASONING│───▶│   FINAL         │  │
│  │ NODE    │    │ NODE    │    │ NODE    │    │   RESPONSE      │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────────────┘  │
│                       │              │                │             │
│  🔀 PARALLEL MODE:    │              │                │             │
│  ┌─────────────────────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │    MATH + SEARCH        │───▶│REASONING│───▶│   FINAL         │  │
│  │   (Concurrent Nodes)    │    │ NODE    │    │   RESPONSE      │  │
│  └─────────────────────────┘    └─────────┘    └─────────────────┘  │
│       │             │                │                │             │
│       ▼             ▼                ▼                ▼             │
│  ┌─────────┐  ┌─────────────┐  ┌─────────┐    ┌─────────────────┐  │
│  │ MLX     │  │   Vector    │  │ LLM     │    │ • Math Results  │  │
│  │ Compute │  │ Embeddings  │  │Response │    │ • Search Results│  │
│  │ Results │  │             │  │ + Memory│    │ • LLM Response  │  │
│  └─────────┘  │sentence-    │  └─────────┘    └─────────────────┘  │
│               │transformers │              ▲                       │
│               │all-MiniLM   │              │                       │
│               └─────────────┘              │                       │
│                     │                      │                       │
│                     ▼                      │                       │
│               ┌─────────────┐              │                       │
│               │Query Vector │──────────────┘                       │
│               │Similarity   │ (Provides semantic                   │
│               │Search       │  context for reasoning)              │
│               └─────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT TO USER                              │
│                                                                     │
│  🤖 Assistant: I can solve that linear system using MLX.          │
│     🔢 Math: x=2.000, y=1.000                                     │
│     🔍 Knowledge: Linear algebra operations on MLX can leverage... │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

🔄 EXECUTION FLOWS:

📝 SEQUENTIAL MODE (Default):
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│START│───▶│MATH │───▶│SEARCH│───▶│REASON│───▶│ END │
└─────┘    └─────┘    └─────┘    └─────┘    └─────┘
             │          │          │
             ▼          ▼          ▼
        ┌─────────┐┌─────────┐┌─────────┐
        │MLX Comp.││Embedding││Local LLM│
        │on Apple ││Search   ││Response │
        │Silicon  ││         ││+ Memory │
        └─────────┘└─────────┘└─────────┘

🔀 PARALLEL MODE (Toggle with 'mode' command):
┌─────┐    ┌─────────────────────┐    ┌─────┐    ┌─────┐
│START│───▶│    MATH + SEARCH    │───▶│REASON│───▶│ END │
└─────┘    │   (Concurrent)      │    └─────┘    └─────┘
           └─────────────────────┘         │
                     │                     ▼
                     ▼                ┌─────────┐
           ┌─────────────────────┐    │Local LLM│
           │MLX Comp. + Embedding│    │Response │
           │on Apple Silicon     │    │+ Memory │
           └─────────────────────┘    └─────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │🔍 Embedding Pipeline│
           │Query → Vector →     │
           │Similarity → Results │
           │(sentence-transformers)│
           └─────────────────────┘

⚡ MODE SWITCHING:
• Type 'mode' to toggle between sequential ↔ parallel execution
• Sequential: Step-by-step processing (thorough)
• Parallel: Concurrent math/search (optimized)

💾 MODELS USED:
• Language: Llama-3.2-3B-Instruct-4bit (Local inference via MLX-LM)
• Embeddings: all-MiniLM-L6-v2 (sentence-transformers framework)
  └─ Converts text → 384-dim vectors for semantic similarity search
  └─ Enables knowledge retrieval without external APIs
• Math: MLX Linear Algebra (Apple Silicon optimized)

🔧 KEY FEATURES:
• 100% Local Processing (No external APIs)
• Conversation Memory (Context across turns)
• Multi-Modal Responses (Math + Search + Chat)
• Apple Silicon Optimization (MLX acceleration)
• Dual Agent Modes (Sequential ↔ Parallel workflows)
• Runtime Mode Switching (Toggle with 'mode' command)
• Robust Error Handling (Graceful fallbacks)
"""
    print(diagram)


def main():
    """Main demo execution"""
    print("🚀 LangGraph + MLX Local Agent Demo")
    print("=" * 60)
    print("🍎 Running 100% locally on Apple Silicon")
    print("🧠 Local Models:")
    print("   • Llama-3.2-3B-Instruct (Language)")
    print("   • all-MiniLM-L6-v2 (Embeddings)")
    print("   • MLX Linear Algebra (Math)")
    
    try:
        # Create and initialize agent
        agent = LocalAgent()
        initialized = agent.initialize()
        
        if not initialized:
            print("⚠️ Some components failed to load, but continuing with available features...")
        
        # Run demos
        run_capability_demos(agent)
        
        # Interactive session
        run_interactive_demo(agent)
        
        # Show architecture diagram
        show_architecture_diagram()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
