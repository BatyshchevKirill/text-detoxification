import torch
import torch.nn as nn

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

torch.manual_seed(12)


class Transformer(nn.Module):
    """
    The transformer model. The implementation is highly similar to the one from
    the video from reference [5]
    """
    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        heads: int,
        enc_layers: int,
        dec_layers: int,
        hidden_dim: int,
        dropout: float,
        max_len: int,
        device: torch.device,
    ):
        """
        Initialize a  Transformer model for sequence-to-sequence tasks.

        :param emb_dim: The dimension of word embeddings.
        :param vocab_size: The size of the vocabulary.
        :param heads: The number of attention heads in the model.
        :param enc_layers: The number of encoder layers.
        :param dec_layers: The number of decoder layers.
        :param hidden_dim: The dimension of the hidden layers.
        :param dropout: The dropout rate.
        :param max_len: The maximum sequence length.
        :param device: The device (CPU or GPU) to run the model.
        """
        super().__init__()
        # Word embeddings
        self.emb = nn.Embedding(vocab_size, emb_dim)

        # Positional embeddings
        self.pos = nn.Embedding(max_len, emb_dim)
        self.device = device

        # Model
        self.transformer = nn.Transformer(
            emb_dim, heads, enc_layers, dec_layers, hidden_dim, dropout
        )

        # Generator
        self.gen = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def mask(self, src):
        """
        Create a mask tensor to identify padding elements in the source sequence.

        :param src: The source sequence tensor.

        :return: A mask tensor of the same shape as src.
        """
        return src.transpose(0, 1) == PAD_IDX

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.

        :param src: The source sequence tensor.
        :param tgt: The target sequence tensor.

        :return: The output tensor produced by the model.
        """
        src_len, batch = src.shape
        tgt_len, _ = tgt.shape

        src_pos = (
            torch.arange(0, src_len).unsqueeze(1).expand(src_len, batch).to(self.device)
        )
        tgt_pos = (
            torch.arange(0, tgt_len).unsqueeze(1).expand(tgt_len, batch).to(self.device)
        )

        emb_src = self.dropout((self.emb(src) + self.pos(src_pos)))
        emb_tgt = self.dropout((self.emb(tgt) + self.pos(tgt_pos)))

        src_mask = self.mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_len, device=self.device
        )

        out = self.transformer(
            emb_src,
            emb_tgt,
            src_key_padding_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return self.gen(out)
