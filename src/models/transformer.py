import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_vocab, output_vocab, model_dim, heads, num_encoder_layers, num_decoder_layers, random_state=None):
        super(Transformer, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.embedding_src = nn.Embedding(input_vocab, model_dim)
        self.embedding_tgt = nn.Embedding(output_vocab, model_dim)

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, heads), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(model_dim, heads), num_decoder_layers)
        self.output_layer = nn.Linear(model_dim, output_vocab)

    def forward(self, src, tgt):
        src_embed = self.embedding_src(src)
        tgt_embed = self.embedding_tgt(tgt)
        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory)
        output_logits = self.output_layer(output)
        return output_logits

