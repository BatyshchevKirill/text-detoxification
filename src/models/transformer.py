import torch
import torch.nn as nn

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

torch.manual_seed(12)


class Transformer(nn.Module):
    def __init__(self, emb_dim, vocab_size, heads,
                 enc_layers, dec_layers, hidden_dim, dropout, max_len, device):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos = nn.Embedding(max_len, emb_dim)
        self.device = device
        self.transformer = nn.Transformer(
            emb_dim,
            heads,
            enc_layers,
            dec_layers,
            hidden_dim,
            dropout
        )
        self.gen = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def mask(self, src):
        return src.transpose(0, 1) == PAD_IDX

    def forward(self, src, tgt):
        src_len, batch = src.shape
        tgt_len, _ = tgt.shape

        src_pos = (torch.arange(0, src_len).unsqueeze(1).expand(src_len, batch).to(self.device))
        tgt_pos = (torch.arange(0, tgt_len).unsqueeze(1).expand(tgt_len, batch).to(self.device))

        emb_src = self.dropout((self.emb(src) + self.pos(src_pos)))
        emb_tgt = self.dropout((self.emb(tgt) + self.pos(tgt_pos)))

        src_mask = self.mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len, device=self.device)

        out = self.transformer(
            emb_src,
            emb_tgt,
            src_key_padding_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return self.gen(out)
