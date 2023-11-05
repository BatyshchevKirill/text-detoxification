from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from transformer import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX
from transformer import Transformer
from ..data.make_dataset import TransformerLoaderCreator, ToxicDataset


def train_transformer(model, save_path, lr, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter("runs/loss_plot")
    step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    best_loss = 1000000000
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        bar = tqdm(train_loader)
        total_loss = 0
        for i, batch in enumerate(bar):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt[:-1])
            out = out.reshape(-1, out.shape[2])
            tgt = tgt[1:].reshape(-1)
            optimizer.zero_grad()
            loss = loss_fn(out, tgt)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if step % 100 == 0:
                writer.add_scalar("Training loss", total_loss / (i + 1), global_step=step)
            step += 1
            bar.set_postfix_str(f"Loss: {total_loss / (i + 1)}")

        model.eval()
        bar = tqdm(val_loader)
        total_loss = 0
        for batch in bar:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt[:-1])
            out = out.reshape(-1, out.shape[2])
            tgt = tgt[1:].reshape(-1)
            loss = loss_fn(out, tgt)
            total_loss += loss.item()

        writer.add_scalar("Val loss", total_loss, global_step=epoch)
        print(f" Val_loss: {total_loss / len(val_loader)}")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_path + 'transformer.pth')


if __name__ == '__main__':
    # TOD: Read from console
    model_name = None # FROM SET, now only transformer is avalable
    save_path = None # from console, string
    dataset_path = None # from console, string
    vocab_path = None # optional, string, if not present: None
    batch_size = 32 # optional from console, int
    random_state = None # optional, from console
    epochs = 20  # optional, from console

    if model_name == 'transformer':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = ToxicDataset(
            dataset_path,
            max_vocab_size=10000,
            vocab_path=vocab_path,
            load_pretrained=(vocab_path is not None)
        )
        train_loader, val_loader = TransformerLoaderCreator(dataset, batch_size, max_len=128, random_state=random_state)()
        model = Transformer(512, len(dataset.vocab), 8, 3, 3, 4, 0.1, 128, device).to(device)
        train_transformer(model, save_path, 4e-3, epochs, train_loader, val_loader)

