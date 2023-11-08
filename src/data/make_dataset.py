import nltk
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

nltk.download('punkt')

# The constants for the vocab
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class ToxicDataset(Dataset):
    """
    Represents the dataset of toxic text
    """

    def __init__(self,
                 filepath: str = "../../data/interim/preprocessed_data.csv",
                 max_vocab_size: int = None,
                 vocab_path: str = "../../data/interim/vocab.pth",
                 load_pretrained: bool = False,
                 train: bool = True
                 ):
        """
        :param filepath: the path to the csv file with the dataset
        :param max_vocab_size: the maximum number of words in the vocabulary
        :param vocab_path: the path where the vocab is stored
        :param load_pretrained: True if we want to load the pretrained vocab
        :param train: True if the dataset is for training, else for validating
        """
        self.train = train
        # Read the data
        df = pd.read_csv(filepath)

        # Cast to lists
        self.toxic = df['toxic'].values.tolist()
        if train:
            self.detoxed = df['detoxed'].values.tolist()

        # Tokenize the words in the sentences
        for i in tqdm(range(len(df))):
            self.toxic[i] = word_tokenize(self.toxic[i])
            if train:
                self.detoxed[i] = word_tokenize(self.detoxed[i])

        if load_pretrained:
            # Load the built vocab
            self.vocab = torch.load(vocab_path)
        else:
            # Create the vocab
            col = (self.toxic + self.detoxed) if train else self.toxic
            self.vocab = build_vocab_from_iterator(
                col,
                specials=special_symbols,
                max_tokens=max_vocab_size
            )
            self.vocab.set_default_index(0)

            # Save the vocab
            torch.save(self.vocab, vocab_path)

    def __getitem__(self, i: int):
        """
        Get an item of the dataset

        :param i: the index of the item
        :return: tuple of translated sentences (toxic_sentence, detoxified_sentence) if train,
                 if test - tuple (toxic_sentence, None)
        """
        return self.vocab(self.toxic[i]), self.vocab(self.detoxed[i]) if self.train else None

    def __len__(self) -> int:
        """
        Number of sentences in the dataset

        :return: number of sentences in the dataset, int
        """
        return len(self.toxic)


class TransformerLoaderCreator:
    """
    Class to create dataloaders for the dataset
    """

    def __init__(
            self,
            dataset: ToxicDataset,
            batch_size: int,
            max_len: int,
            test_size: float = 0.2,
            random_state: int = None
    ):
        """
        :param dataset: the dataset instance to create loaders from
        :param batch_size: the size of the batch
        :param max_len: the maximal length of the sentence
        :param test_size: the proportion of test data, if 1, then test mode is used
        :param random_state: the number to seed to random generators
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state
        self.test_size = test_size

        # Calculate number of train and test samples
        num_samples = len(dataset)
        num_test_samples = int(num_samples * self.test_size)
        num_train_samples = num_samples - num_test_samples

        # If train mode, split the data and create train data loader
        if test_size != 1:
            # Split the data
            train_dataset, test_dataset = random_split(
                dataset,
                [num_train_samples, num_test_samples],
                generator=torch.Generator().manual_seed(self.random_state)
            )

            # Create a train data loader with manual seed
            sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(random_state))
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn,
            )
        else:
            test_dataset = dataset

        # Create the validation (test) data loader
        self.val_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        """
        The function that collates batches, preparing them to feeding to the NN

        :param batch: the batch to prepare
        :return: the prepared batches
        """

        if self.test_size != 1:
            # If in training mode, prepare target sentences
            src_texts, tgt_texts = zip(*batch)
            tgt_texts = list(tgt_texts)

            # Cut too long sentences
            tgt_tokens = [text[:self.max_len - 2] for text in tgt_texts]

            # Pad too short sentences, add special symbols
            tgt_padded = [
                [BOS_IDX] + tokens + [EOS_IDX] + [PAD_IDX] * (self.max_len - len(tokens) - 2)
                for tokens in tgt_tokens
            ]

            # Convert to tensors
            tgt_tensors = torch.LongTensor(tgt_padded).transpose(1, 0)
        else:
            src_texts, _ = zip(*batch)

        # Prepare input sentences
        src_texts = list(src_texts)

        # Cut too long sentences
        src_tokens = [text[:self.max_len - 2] for text in src_texts]

        # Pad too short sentences, add special symbols
        src_padded = [
            [BOS_IDX] + tokens + [EOS_IDX] + [PAD_IDX] * (self.max_len - len(tokens) - 2)
            for tokens in src_tokens
        ]

        # Convert to tensor
        src_tensors = torch.LongTensor(src_padded).transpose(1, 0)

        if self.test_size != 1:
            # Return train batch
            return src_tensors, tgt_tensors

        # Return test batch
        return src_tensors

    def __call__(self):
        """
        :return: train and validation data loaders if in train mode, else test data loader
        """
        if self.test_size == 1:
            return self.train_loader
        return self.train_loader, self.val_loader
