import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.data.make_dataset import ToxicDataset, TransformerLoaderCreator
from src.models.preprocess import file_creatable_path, file_path
from src.models.transformer import PAD_IDX, Transformer


def train_transformer(
        model,
        save_path: str,
        lr: float,
        epochs: int,
        train_loader,
        val_loader
):
    """
    Train a Transformer model.

    :param model: The Transformer model to train.
    :param save_path: Path to save the trained model checkpoint.
    :param lr: Learning rate for training.
    :param epochs: Number of training epochs.
    :param train_loader: DataLoader for the training data.
    :param val_loader: DataLoader for the validation data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter("runs/loss_plot")
    step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    best_loss = 1000000000

    for epoch in range(epochs):
        # Train loop
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        bar = tqdm(train_loader)
        total_loss = 0
        for i, batch in enumerate(bar):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            # Process the inputs
            out = model(src, tgt[:-1])
            out = out.reshape(-1, out.shape[2])
            tgt = tgt[1:].reshape(-1)

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(out, tgt)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # Save to tensorboard
            if step % 100 == 0:
                writer.add_scalar(
                    "Training loss", total_loss / (i + 1), global_step=step
                )
            step += 1
            bar.set_postfix_str(f"Loss: {total_loss / (i + 1)}")

        # Validation loop
        model.eval()
        bar = tqdm(val_loader)
        total_loss = 0
        for batch in bar:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            # Process the data
            out = model(src, tgt[:-1])
            out = out.reshape(-1, out.shape[2])
            tgt = tgt[1:].reshape(-1)

            # Compute the loss
            loss = loss_fn(out, tgt)
            total_loss += loss.item()

        # Save the loss to tensorboard
        writer.add_scalar("Val loss", total_loss, global_step=epoch)
        print(f" Val_loss: {total_loss / len(val_loader)}")

        # Save the best model
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_path + "transformer.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model parser")
    parser.add_argument("model_name", choices=["transformer"])
    parser.add_argument("save_path", type=file_creatable_path)
    parser.add_argument("dataset_path", type=file_path)
    parser.add_argument("--vocab-path", default=None, type=file_path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=float, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    (
        model_name,
        save_path,
        dataset_path,
        vocab_path,
        batch_size,
        random_state,
        epochs,
    ) = (
        args.model_name,
        args.save_path,
        args.dataset_path,
        args.vocab_path,
        args.batch_size,
        args.random_state,
        args.epochs,
    )

    if model_name == "transformer":
        # Choose the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a dataset
        dataset = ToxicDataset(
            dataset_path,
            max_vocab_size=10000,
            vocab_path=vocab_path,
            load_pretrained=(vocab_path is not None),
        )
        # Create dataloaders
        train_loader, val_loader = TransformerLoaderCreator(
            dataset, batch_size, max_len=128, random_state=random_state
        )()
        # Initialize the model
        model = Transformer(512, len(dataset.vocab), 8, 3, 3, 4, 0.1, 128, device).to(
            device
        )
        # Train the model
        train_transformer(model, save_path, 4e-3, epochs, train_loader, val_loader)
