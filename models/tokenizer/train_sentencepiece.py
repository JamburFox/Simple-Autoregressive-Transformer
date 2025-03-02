import os
import argparse
import sentencepiece as sp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the tokenizer model.')
    parser.add_argument('--input', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt"), help='Input to train on')
    parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"), help='Output location + file prefix to save as')
    parser.add_argument('--vocab_size', type=int, default=20000, help='The vocab size')
    args = parser.parse_args()

    special_tokens = ['<sep>', '<user>', '<bot>']
    tokenizer = sp.SentencePieceTrainer.Train(input=args.input, model_prefix=args.output, vocab_size=args.vocab_size, pad_id=3, remove_extra_whitespaces=False, add_dummy_prefix=False, user_defined_symbols=special_tokens)