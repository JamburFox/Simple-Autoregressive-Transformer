import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
import argparse

from utils import count_parameters
from model import Transformer
from tokenizer import Tokenizer
from dataloader import get_data_loader
from model_manager import load_transformer_model, save_transformer_model

def train_step(data_loader: torch.utils.data.DataLoader, model: Transformer, optimizer: torch.optim.Optimizer, device: str):
    running_loss = 0

    model.train()
    for batch, (text_data, text_pad_mask) in enumerate(data_loader):
        text_data, text_pad_mask = text_data.to(device), text_pad_mask.to(device)

        #shift data so that the in_text is the initial tokens and that tgt_text is the next predicted token in the sequence
        in_text = text_data[:, :-1]#shifts the data to the left ignoring the last index
        in_mask = text_pad_mask[:, :-1]
        tgt_text = text_data[:, 1:]#shifts the data to the right ignoring the first input
        tgt_mask = text_pad_mask[:, 1:]

        out = model(in_text)

        outputs = out[:, :].reshape(-1, model.vocab_size)# Reshape to [batch_size * seq, vocab_size]
        targets = tgt_text[:, :].reshape(-1)# Reshape to [batch_size * seq]

        loss = criterion(outputs, targets)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        #clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if batch % 1 == 0:
            print(f"\033[K=== Train Batch {batch+1} / {len(data_loader)} ({(batch+1)/len(data_loader)*100:.2f}%) ===", end="\r")
            #print(f"=== Train Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_loss = running_loss / len(data_loader)
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the transformer model.')
    parser.add_argument('--dataset', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "dataset.hdf5"), help='location of the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--scheduler_step', type=int, default=5, help='batches before adjusting learning rate')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='learning rate multiplier for each scheduler step')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    torch.cuda.empty_cache()

    tokenizer = Tokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tokenizer", "tokenizer.model"))

    model = load_transformer_model(tokenizer).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    train_loader = get_data_loader(args.dataset, args.batch_size, tokenizer)

    print(f"Parameters: {count_parameters(model):,}")

    for epoch in range(0, args.epochs):
        train_loss = train_step(train_loader, model, optimizer, args.device)
        print(f"=== Epoch: {epoch+1} / {args.epochs} | Train_loss: {train_loss:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}===")

        save_transformer_model(model)
        scheduler.step()
