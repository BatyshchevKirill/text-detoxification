from src.models.transformer import Transformer, BOS_IDX, EOS_IDX
import torch
import nltk
from nltk import word_tokenize

nltk.download("punkt")


class  TransformerPredictor:
    def __init__(
            self,
            model_config,
            model_path: str = "models/transformer_checkpoints/transformer.pth",
            vocab_path: str = "data/interim/vocab.pth",
            max_len: int = 128
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Transformer(*model_config, self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.vocab = torch.load(vocab_path)
        self.itos = self.vocab.get_itos()
        self.max_len = max_len

    def __call__(self, text):
        sent = word_tokenize(text)
        sent = [BOS_IDX] + self.vocab(sent) + [EOS_IDX]
        sent = torch.LongTensor(sent).to(self.device).view(-1, 1)

        result = torch.LongTensor([[BOS_IDX]]).to(self.device)

        for _ in range(self.max_len):
            out = self.model(sent, result)
            _, pred = torch.max(out, dim=2)
            nxt = pred[-1]
            if nxt[0].item() == EOS_IDX:
                break
            result = torch.cat((result, nxt.unsqueeze(0)), dim=0)

        result = result[1:].view(-1).detach().cpu().tolist()

        translated_sent = []
        for token in result:
            if token > EOS_IDX:
                translated_sent.append(self.itos[token])

        translated_sent = " ".join(translated_sent).replace(" ,", ",").replace(" .", ".").replace(" '", "'")
        return translated_sent


