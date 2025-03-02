import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import generate_square_subsequent_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim

        self.query_layer = nn.Linear(input_dim, num_heads * attention_dim)
        self.key_layer = nn.Linear(input_dim, num_heads * attention_dim)
        self.value_layer = nn.Linear(input_dim, num_heads * attention_dim)

        self.fc_out = nn.Linear(num_heads * attention_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()

        queries = self.query_layer(x).view(batch_size, seq_length, self.num_heads, self.attention_dim)
        keys = self.key_layer(x).view(batch_size, seq_length, self.num_heads, self.attention_dim)
        values = self.value_layer(x).view(batch_size, seq_length, self.num_heads, self.attention_dim)

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 1, 3)

        #Compute attention scores using the dot product between queries and keys
        attention_scores = torch.matmul(queries, keys) / (self.attention_dim ** 0.5)#[batch, heads, seq, seq]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)#Apply softmax to obtain attention weights
        attended_values = torch.matmul(attention_weights, values)

        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        attended_values = attended_values.view(batch_size, seq_length, -1)#Concatenate heads

        output = self.fc_out(attended_values)#Compute the weighted sum of values
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim=embedding_dim, dropout=dropout, max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)#embedded = [batch, sequence, embedding_dim]

        #positional encoding
        embedded = embedded.permute(1, 0, 2)#embedded = [sequence, batch, embedding_dim]
        embedded = self.positional_encoding(embedded)#embedded = [sequence, batch, embedding_dim] (with positional encoding)
        embedded = embedded.permute(1, 0, 2)#embedded = [batch, sequence, embedding_dim]

        return embedded

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int, feedforward_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)

        self.masked_attention = MultiHeadAttention(input_dim=embedding_dim, attention_dim=attention_dim, num_heads=num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=feedforward_dim),
            nn.ReLU(),
            nn.Linear(in_features=feedforward_dim, out_features=embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: #x = [batch, sequence]
        #layer norm 1
        x_norm1 = self.layer_norm1(x)#x_norm1 = [batch, sequence, embedding_dim]

        #create square mask so that future values are not attended to
        s_mask = generate_square_subsequent_mask(x.shape[1]).to(x.device) #[seq, seq]

        #masked multi-head attention (masking currently not implemented)
        x_attended = self.masked_attention(x_norm1, s_mask)#x_attended = [batch, sequence, embedding_dim]

        #residual connection 1
        x_residual1 = x_attended + x#x_residual1 = [batch, sequence, embedding_dim]

        #layer norm 2
        x_norm2 = self.layer_norm2(x_residual1)#x_norm2 = [batch, sequence, embedding_dim]

        #feed-forward network
        x_feedforward = self.feed_forward(x_norm2)#x_feedforward = [batch, sequence, out_features]

        #residual connection 2
        x_residual2 = x_feedforward + x_residual1#x_residual1 = [batch, sequence, embedding_dim]

        return x_residual2

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, attention_dim: int, num_heads: int, feedforward_dim: int, num_decoder_layers: int):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = PositionalEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(embedding_dim=embedding_dim, attention_dim=attention_dim, num_heads=num_heads, feedforward_dim=feedforward_dim) for _ in range(num_decoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: #x = [batch, sequence]
        #embedding + positional encoding
        x_embedded = self.embedding(x)#embedded = [batch, sequence, embedding_dim]

        #decoder layers
        for decoder in self.decoders:
            x_embedded = decoder(x_embedded)#decoder_out = [batch, sequence, embedding_dim]

        #layer norm
        x_norm = self.layer_norm(x_embedded)#x_norm = [batch, sequence, embedding_dim]

        #linear
        logits = self.linear(x_norm)#x_norm = [batch, sequence, vocab_size]

        return logits
    
    def infer_one_step(self, x: torch.Tensor):
        logits = self.forward(x)#[batch, seq, vocab]
        last_logits = logits[:,-1,:]#[batch, vocab]
        predicted_tokens = token_selection(last_logits, top_p=0.5, temperature=1)#[batch]
        return predicted_tokens
    
    def infer(self, x: torch.Tensor, eos_token: int, max_length: int = 100):
        model_input = x
        for _ in range(0, max_length):
            tokens = self.infer_one_step(model_input)

            if torch.all(tokens == eos_token):
                break

            model_input = torch.cat((model_input, tokens), dim=1)

        return model_input

def token_selection(logits: torch.Tensor, top_p: int = 0.5, temperature: float = 1.0):#[1,5000]
    # Apply temperature scaling to the logits
    logits = logits / temperature

    probabilities = torch.softmax(logits, dim=-1)# Compute probabilities from logits so they are based on values that add up to 1
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)# Compute the cumulative sum of probabilities

    cutoff_mask = cumulative_probabilities <= top_p#converts into mask where true is valid and false is not considered
    cutoff_index = max(cutoff_mask.sum().item(), 1)#return the max cutoff index for each batch while keeping at minimum 1 cutoff index

    top_p_probabilities = sorted_probabilities[:, :cutoff_index]
    top_p_indices = sorted_indices[:, :cutoff_index]

    top_p_probabilities /= top_p_probabilities.sum()#normalize probabilities to the sum of 1

    next_tokens = torch.multinomial(top_p_probabilities, 1)#select tokens to use
    selected_indices = torch.gather(top_p_indices, 1, next_tokens)#select the indicies based on next_token for each batch

    return selected_indices