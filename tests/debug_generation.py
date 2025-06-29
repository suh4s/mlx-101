#!/usr/bin/env python3
"""
Simple test for LLM generation to debug the issue
"""

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

try:
    from mlx_lm import load, generate
    
    print("🧪 Testing LLM Generation...")
    
    # Load model
    print("📥 Loading Llama-3.2-3B...")
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    print("✅ Model loaded")
    
    # Test with simple prompt
    test_message = "What is Apple Silicon?"
    
    # Try the Llama chat format
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{test_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    print(f"🔄 Generating response for: '{test_message}'")
    print(f"📝 Prompt length: {len(prompt)} characters")
    
    try:
        # Test with max_tokens parameter
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False
        )
        print(f"✅ Generated response:")
        print(f"   Length: {len(response)} characters")
        print(f"   Content: '{response}'")
        
        # Check if truncated
        if not response.strip().endswith(('.', '!', '?')):
            print("⚠️  Response appears truncated")
        else:
            print("✅ Response appears complete")
            
    except Exception as e:
        print(f"❌ Generation with params failed: {e}")
        
        # Try without parameters
        try:
            response = generate(model, tokenizer, prompt)
            print(f"✅ Simple generation worked:")
            print(f"   Content: '{response}'")
        except Exception as e2:
            print(f"❌ Simple generation also failed: {e2}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
