import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
import nltk
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle
from tqdm import tqdm

nltk.download('punkt')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class ToxicDataset(Dataset):
    def __init__(self, filepath="../../data/interim/preprocessed_data.csv", max_vocab_size=None,
                 vocab_path="../../data/interim/vocab.pth",
                 load_pretrained=False):
        df = pd.read_csv(filepath)
        self.toxic = df['toxic'].values.tolist()
        self.detoxed = df['detoxed'].values.tolist()

        for i in tqdm(range(len(df))):
            self.toxic[i] = word_tokenize(self.toxic[i])
            self.detoxed[i] = word_tokenize(self.detoxed[i])

        if load_pretrained:
            self.vocab = torch.load(vocab_path)
        else:
            self.vocab = build_vocab_from_iterator(self.toxic + self.detoxed,
                                                   specials=special_symbols, max_tokens=max_vocab_size)
            self.vocab.set_default_index(0)
            torch.save(self.vocab, vocab_path)

    def __getitem__(self, i: int):
        return self.vocab(self.toxic[i]), self.vocab(self.detoxed[i])

    def __len__(self):
        return len(self.toxic)


class TransformerLoaderCreator:
    def __init__(self, dataset, batch_size, max_len, test_size=0.2, random_state=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state

        num_samples = len(dataset)
        num_test_samples = int(num_samples * self.test_size)
        num_train_samples = num_samples - num_test_samples
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples],
                                                   generator=torch.Generator().manual_seed(self.random_state))

        sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(random_state))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

        self.val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        src_texts, tgt_texts = zip(*batch)
        src_texts = list(src_texts)
        tgt_texts = list(tgt_texts)

        src_tokens = [text[:self.max_len] for text in src_texts]
        src_padded = [tokens + PAD_IDX * (self.max_len - len(tokens)) for tokens in src_tokens]
        src_tensors = torch.LongTensor(src_padded)

        tgt_tokens = [text[:self.max_len] for text in tgt_texts]
        tgt_padded = [tokens + PAD_IDX * (self.max_len - len(tokens)) for tokens in tgt_tokens]
        tgt_tensors = torch.LongTensor(tgt_padded)

        return src_tensors, tgt_tensors

    def __call__(self):
        return self.train_loader, self.val_loader
