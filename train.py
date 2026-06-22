#!/usr/bin/env python3
"""
Macha Training Script
"""
import sys, os, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macha.model.transformer import MachaModel, MachaTokenizer
from macha.training.trainer import MachaTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Macha AI")
    parser.add_argument('--data', default='data/training_data.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--save_path', default='checkpoints/macha_model.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print("Training Macha AI...")
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = MachaTokenizer(vocab_size=args.vocab_size)
    all_texts = [msg['content'] for conv in data for msg in conv['messages']]
    tokenizer.fit(all_texts)

    model = MachaModel(vocab_size=len(tokenizer.word2idx), d_model=args.d_model, num_heads=8, num_layers=4, device=args.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = MachaTrainer(model, tokenizer, lr=args.lr, device=args.device)
    trainer.train(data, args.epochs, args.batch_size, args.save_path)
    print("Training complete!")

if __name__ == "__main__":
    main()
