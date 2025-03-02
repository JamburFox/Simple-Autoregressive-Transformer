import torch
import os
import argparse

from tokenizer import Tokenizer
from model_manager import load_transformer_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the transformer model.')
    parser.add_argument('--text_input', nargs='+', type=str, default=["hello world"], help='the device to use')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    tokenizer = Tokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tokenizer", "tokenizer.model"))
    model = load_transformer_model(tokenizer).to(args.device)

    encoded, mask = tokenizer.encode_batch(args.text_input)
    encoded, mask = encoded.to(args.device), mask.to(args.device)

    model_out = model.infer(encoded, tokenizer.eos_id, 100)

    decoded = tokenizer.decode_batch(model_out)
    for i, text in enumerate(decoded):
        print(f"[{i}] Decoded:", text)