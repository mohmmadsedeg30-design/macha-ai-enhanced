"""
Macha Trainer - Training + Learning from mistakes
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json, os
from tqdm import tqdm
from typing import List, Dict
import random


class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        for conv in conversations:
            for i in range(len(conv['messages']) - 1):
                inp = self.tokenizer.encode(conv['messages'][i]['content'], max_len)
                tgt = self.tokenizer.encode(conv['messages'][i+1]['content'], max_len)
                self.samples.append((inp, tgt))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx][0]), torch.tensor(self.samples[idx][1])


class MachaTrainer:
    def __init__(self, model, tokenizer, lr=1e-4, wd=0.01, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-8, weight_decay=wd)
        self.scheduler = None
        self.best_loss = float('inf')

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, num_batches = 0, 0
        for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits, loss = self.model(inputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, train_data, epochs=10, batch_size=16, save_path='checkpoints/macha_model.pt'):
        print(f"Training Macha for {epochs} epochs...")
        random.shuffle(train_data)
        dataset = ConversationDataset(train_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            loss = self.train_epoch(dataloader)
            print(f"Loss: {loss:.4f}")
            if loss < self.best_loss:
                self.best_loss = loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.model.save_checkpoint(save_path, self.tokenizer, self.optimizer, epoch, loss)
                print("Best model saved!")
        return {'best_loss': self.best_loss}

    def learn_from_mistakes(self, feedback_data, epochs=5, batch_size=8, save_path='checkpoints/macha_model.pt'):
        print(f"Learning from {len(feedback_data)} corrections...")
        return self.train(feedback_data, epochs, batch_size, save_path)
