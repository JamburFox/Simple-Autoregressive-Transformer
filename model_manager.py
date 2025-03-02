import os
import torch

from model import Transformer
from tokenizer import Tokenizer

EMBEDDING_DIM = 512
ATTENTION_DIM = 256
NUM_HEADS = 2
FEEDFORWARD_DIM = 2048
NUM_DECODER_LAYERS = 4

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(SAVE_PATH, "transformer_model.pt")

def load_transformer_model(tokenizer: Tokenizer, file_path: str = DEFAULT_MODEL_PATH) -> Transformer:
    model = Transformer(tokenizer.vocab_size, EMBEDDING_DIM, ATTENTION_DIM, NUM_HEADS, FEEDFORWARD_DIM, NUM_DECODER_LAYERS)
    if file_path is not None:
        try:
            model.load_state_dict(torch.load(f=file_path))
        except:
            print(f"Unable to load model from {file_path}!")
    return model

def save_transformer_model(model: Transformer, save_path: str = DEFAULT_MODEL_PATH):
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(obj=model.state_dict(), f=save_path)
    except Exception as e:
        print(f"Unable to save model to {save_path}! {e}")