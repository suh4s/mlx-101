#!/usr/bin/env python3
"""
Test LLM generation directly
"""

try:
    from mlx_lm import load, generate
    
    print("🧪 Testing MLX-LM Generation...")
    print("📥 Loading model (this may take a moment)...")
    
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    print("✅ Model loaded")
    
    # Test simple generation
    prompt = "User: What is 2+2?\nAssistant:"
    
    print("🔄 Testing generation...")
    try:
        # Try newer API
        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
        print(f"✅ Generation successful: {response.strip()}")
    except TypeError as e:
        print(f"❌ New API failed: {e}")
        try:
            # Try simpler API
            response = generate(model, tokenizer, prompt)
            print(f"✅ Simple API works: {response.strip()}")
        except Exception as e2:
            print(f"❌ Simple API also failed: {e2}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
