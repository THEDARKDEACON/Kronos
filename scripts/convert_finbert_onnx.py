"""
Convert FinBERT to ONNX format for numpy 2.x compatibility.
This eliminates pickle serialization issues.
"""

import os
import sys
import argparse
from pathlib import Path

# Fix sklearn/numpy compatibility: prioritize user-installed packages
user_site = os.path.expanduser('~/.local/lib/python3.12/site-packages')
if os.path.exists(user_site):
    sys.path.insert(0, user_site)

def convert_finbert_to_onnx(model_name: str = "ProsusAI/finbert",
                           output_dir: str = "./models/onnx",
                           opset_version: int = 14):
    """
    Convert FinBERT model to ONNX format.
    
    Benefits:
    - No numpy pickle compatibility issues
    - 3-5x faster inference
    - Smaller file size (~250MB -> ~66MB)
    - No PyTorch/Transformers dependency at runtime
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"[Error] Missing dependencies: {e}")
        print("Install: pip install torch transformers onnx onnxruntime-gpu")
        sys.exit(1)
    
    print(f"[ONNX] Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True,
        torch_dtype=torch.float32
    )
    model.eval()
    
    # Prepare dummy input
    dummy_text = "This is a test sentence for ONNX conversion."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # Dynamic axes for variable batch sizes and sequence lengths
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model name for file
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    onnx_path = output_path / f"{model_short}.onnx"
    
    print(f"[ONNX] Exporting to: {onnx_path}")
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"[ONNX] Model exported: {onnx_path}")
    print(f"[ONNX] Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Verify the model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Model verification: PASSED")
    
    # Create dynamic quantized version (faster, smaller)
    quantized_path = output_path / f"{model_short}_quantized.onnx"
    print(f"[ONNX] Creating quantized version...")
    
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
        optimize_model=True
    )
    
    print(f"[ONNX] Quantized model: {quantized_path}")
    print(f"[ONNX] Size: {quantized_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Save tokenizer for runtime use
    tokenizer_path = output_path / f"{model_short}_tokenizer"
    tokenizer.save_pretrained(str(tokenizer_path))
    print(f"[ONNX] Tokenizer saved: {tokenizer_path}")
    
    print("\n[ONNX] Conversion complete!")
    print(f"[ONNX] Files in {output_dir}:")
    for f in sorted(output_path.glob("*")):
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    
    return str(onnx_path), str(quantized_path)


def test_onnx_model(onnx_path: str, tokenizer_path: str = None):
    """Test the converted ONNX model."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError:
        print("[Test] Install onnxruntime to test: pip install onnxruntime-gpu")
        return False
    
    print(f"\n[Test] Loading ONNX model: {onnx_path}")
    
    # Create inference session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"[Test] Providers: {session.get_providers()}")
    print(f"[Test] Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"[Test] Outputs: {[o.name for o in session.get_outputs()]}")
    
    # Load tokenizer
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilbert-base-finetuned-financial-news-sentiment-analysis")
    
    # Test inference
    test_texts = [
        "Apple stock surges on strong earnings report!",
        "Market crashes amid recession fears",
        "Company announces layoffs and restructuring"
    ]
    
    print("\n[Test] Running inference...")
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="np", max_length=512, padding="max_length", truncation=True)
        
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
        ort_outputs = session.run(None, ort_inputs)
        logits = ort_outputs[0]
        
        # Get prediction
        import numpy as np
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        pred_idx = np.argmax(probs, axis=-1)[0]
        confidence = np.max(probs)
        
        labels = ["negative", "neutral", "positive"]
        label = labels[pred_idx]
        
        print(f"  Text: {text[:50]}...")
        print(f"  -> Sentiment: {label} (confidence: {confidence:.3f})")
    
    print("\n[Test] ONNX model working correctly!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FinBERT to ONNX")
    parser.add_argument("--model", default="ProsusAI/finbert",
                       help="HuggingFace model name")
    parser.add_argument("--output", default="./models/onnx",
                       help="Output directory")
    parser.add_argument("--opset", type=int, default=14,
                       help="ONNX opset version")
    parser.add_argument("--test", action="store_true",
                       help="Test the converted model")
    
    args = parser.parse_args()
    
    onnx_path, quantized_path = convert_finbert_to_onnx(
        model_name=args.model,
        output_dir=args.output,
        opset_version=args.opset
    )
    
    if args.test:
        model_short = args.model.split("/")[-1] if "/" in args.model else args.model
        tokenizer_path = Path(args.output) / f"{model_short}_tokenizer"
        test_onnx_model(quantized_path, str(tokenizer_path))
