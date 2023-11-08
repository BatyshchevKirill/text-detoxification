import nltk
import torch
from nltk import word_tokenize

from src.models.transformer import BOS_IDX, EOS_IDX, Transformer

nltk.download("punkt")


class TransformerPredictor:
    """
    This is a wrapper class that uses a transformer model of Transformer class to
    perform a prediction on a sentence
    """

    def __init__(
        self,
        model_config,
        model_path: str = "models/transformer_checkpoints/transformer.pth",
        vocab_path: str = "data/interim/vocab.pth",
        max_len: int = 128,
    ):
        """
        :param model_config: tuple containing model parameters
        :param model_path: path to model checkpoint
        :param vocab_path: path to vocab
        :param max_len: the maximal length of the sentence
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Transformer(*model_config, self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.vocab = torch.load(vocab_path)
        self.itos = self.vocab.get_itos()
        self.max_len = max_len

    def __call__(self, text: str) -> str:
        """
        Translate the input text to another language using the trained Transformer model.

        :param text: The input text to be translated.

        :return: The translated text.
        """
        # Tokenize the sentence
        sent = word_tokenize(text)

        # Convert to numbers
        sent = [BOS_IDX] + self.vocab(sent) + [EOS_IDX]

        # Convert to tensor
        sent = torch.LongTensor(sent).to(self.device).view(-1, 1)

        # Create a tensor for output
        result = torch.LongTensor([[BOS_IDX]]).to(self.device)

        for _ in range(self.max_len):
            # Get the prediction
            out = self.model(sent, result)
            _, pred = torch.max(out, dim=2)
            nxt = pred[-1]

            # Break if reached end-of-string token
            if nxt[0].item() == EOS_IDX:
                break
            result = torch.cat((result, nxt.unsqueeze(0)), dim=0)

        # Get the result list
        result = result[1:].view(-1).detach().cpu().tolist()

        # Translate the sentence ignoring special symbols
        translated_sent = []
        for token in result:
            if token > EOS_IDX:
                translated_sent.append(self.itos[token])

        # Prettify the result
        translated_sent = (
            " ".join(translated_sent)
            .replace(" ,", ",")
            .replace(" .", ".")
            .replace(" '", "'")
        )
        return translated_sent
