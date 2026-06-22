"""
Macha Transformer Model
Small transformer for incremental learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from typing import Optional, Tuple, List


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)
        return self.W_O(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class MachaTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, "<user>": 4, "<bot>": 5}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = {}

    def fit(self, texts: List[str]):
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_count[word] = self.word_count.get(word, 0) + 1
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - len(self.word2idx)]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def _tokenize(self, text: str) -> List[str]:
        import re
        tokens = re.findall(r'[؀-ۿ]+|[a-zA-Z]+|[0-9]+|[^\w\s]', text.lower())
        return tokens

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        words = self._tokenize(text)
        ids = [self.word2idx.get("<sos>", 1)]
        for word in words[:max_len - 2]:
            ids.append(self.word2idx.get(word, self.word2idx["<unk>"]))
        ids.append(self.word2idx.get("<eos>", 2))
        while len(ids) < max_len:
            ids.append(self.word2idx["<pad>"])
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx in [0, 1, 2]:
                continue
            word = self.idx2word.get(idx, "<unk>")
            if word not in ["<pad>", "<sos>", "<eos>", "<unk>", "<user>", "<bot>"]:
                words.append(word)
        return " ".join(words)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': {str(k): v for k, v in self.idx2word.items()}, 'vocab_size': self.vocab_size}, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            self.vocab_size = data['vocab_size']


class MachaModel(nn.Module):
    def __init__(self, vocab_size: int = 10000, d_model: int = 256, num_heads: int = 8, num_layers: int = 4, d_ff: int = 512, max_seq_len: int = 128, dropout: float = 0.1, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.to(device)

    def make_causal_mask(self, seq_len: int):
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask.to(self.device)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        mask = self.make_causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 40) -> str:
        self.eval()
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=self.device)
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(input_ids_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.word2idx.get("<eos>", 2):
                break
        return tokenizer.decode(input_ids[0].tolist())

    def save_checkpoint(self, path: str, tokenizer, optimizer=None, epoch=0, loss=0):
        checkpoint = {'model_state_dict': self.state_dict(), 'vocab_size': self.vocab_size, 'd_model': self.d_model, 'num_layers': len(self.blocks), 'epoch': epoch, 'loss': loss}
        torch.save(checkpoint, path)
        tokenizer.save(path.replace('.pt', '_tokenizer.json'))
        print(f"Saved model to: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {path}")
        return checkpoint


if __name__ == "__main__":
    print("Testing Macha Model")
    model = MachaModel(vocab_size=1000, d_model=64, num_heads=4, num_layers=2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
