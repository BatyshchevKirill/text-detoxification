import nltk
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

nltk.download('punkt')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class ToxicDataset(Dataset):
    def __init__(self, filepath="../../data/interim/preprocessed_data.csv", max_vocab_size=None,
                 vocab_path="../../data/interim/vocab.pth",
                 load_pretrained=False,
                 train=True):
        self.train = train
        df = pd.read_csv(filepath)
        self.toxic = df['toxic'].values.tolist()
        if train:
            self.detoxed = df['detoxed'].values.tolist()

        for i in tqdm(range(len(df))):
            self.toxic[i] = word_tokenize(self.toxic[i])
            if train:
                self.detoxed[i] = word_tokenize(self.detoxed[i])

        if load_pretrained:
            self.vocab = torch.load(vocab_path)
        else:
            col = self.toxic + self.detoxed if train else self.toxic
            self.vocab = build_vocab_from_iterator(col,
                                                   specials=special_symbols, max_tokens=max_vocab_size)
            self.vocab.set_default_index(0)
            torch.save(self.vocab, vocab_path)

    def __getitem__(self, i: int):
        return self.vocab(self.toxic[i]), self.vocab(self.detoxed[i]) if self.train else None

    def __len__(self):
        return len(self.toxic)


class TransformerLoaderCreator:
    def __init__(self, dataset, batch_size, max_len, test_size=0.2, random_state=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state
        self.test_size = test_size

        num_samples = len(dataset)
        num_test_samples = int(num_samples * self.test_size)
        num_train_samples = num_samples - num_test_samples
        if test_size != 1:
            train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples],
                                                       generator=torch.Generator().manual_seed(self.random_state))
            sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(random_state))
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn,
            )
        else:
            test_dataset = dataset
        self.val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        if self.test_size != 1:
            src_texts, tgt_texts = zip(*batch)
            tgt_texts = list(tgt_texts)
            tgt_tokens = [text[:self.max_len - 2] for text in tgt_texts]
            tgt_padded = [
                [BOS_IDX] + tokens + [EOS_IDX] + [PAD_IDX] * (self.max_len - len(tokens) - 2)
                for tokens in tgt_tokens
            ]
            tgt_tensors = torch.LongTensor(tgt_padded).transpose(1, 0)
        else:
            src_texts, _ = zip(*batch)
        src_texts = list(src_texts)
        src_tokens = [text[:self.max_len - 2] for text in src_texts]
        src_padded = [
            [BOS_IDX] + tokens + [EOS_IDX] + [PAD_IDX] * (self.max_len - len(tokens) - 2)
            for tokens in src_tokens
        ]
        src_tensors = torch.LongTensor(src_padded).transpose(1, 0)
        if self.test_size != 1:
            return src_tensors, tgt_tensors
        return src_tensors

    def __call__(self):
        if self.test_size == 1:
            return self.train_loader
        return self.train_loader, self.val_loader
