import torch
import sentencepiece as spm

class Tokenizer():
    def __init__(self, model_path: str):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

        self.vocab_size = self.tokenizer.vocab_size()
        self.unk_id = self.tokenizer.PieceToId('<unk>')
        self.bos_id = self.tokenizer.PieceToId('<s>')
        self.eos_id = self.tokenizer.PieceToId('</s>')
        self.pad_id = self.tokenizer.PieceToId('<pad>')
        self.sep_id = self.tokenizer.PieceToId('<sep>')
        self.user_id = self.tokenizer.PieceToId('<user>')
        self.bot_id = self.tokenizer.PieceToId('<bot>')
        self.special_tokens = [self.unk_id, self.bos_id, self.eos_id, self.pad_id, self.sep_id, self.user_id, self.bot_id]

    def filter_special_tokens(self, encoded_seq):
        return [token for token in encoded_seq if token not in self.special_tokens]

    def encode_batch(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_text = self.tokenizer.EncodeAsIds(text)
        batch = len(encoded_text)
        max_len = len(max(encoded_text, key=len))

        tokens = torch.zeros(batch, max_len, dtype=torch.long) + self.pad_id
        tokens_pad_mask = torch.zeros(batch, max_len, dtype=torch.float)
        for i, seq in enumerate(encoded_text):
            seq_len = len(seq)
            tokens[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            tokens_pad_mask[i, :seq_len] = 1

        return tokens, tokens_pad_mask
    
    def decode_batch(self, tokens: torch.Tensor, remove_special: bool = False) -> list[str]:
        sequences = []

        batch = tokens.shape[0]
        for i in range(batch):
            seq = tokens[i, :].tolist()
            if remove_special:
                seq = self.filter_special_tokens(seq)
            sequences.append(seq)

        decoded_text = self.tokenizer.DecodePieces(sequences)
        return decoded_text