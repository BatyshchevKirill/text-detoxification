import torch
from transformers import BertForMaskedLM, BertTokenizer


class BaselineModel:
    def __init__(self, toxic_words_path='../data/interim/toxic_words.txt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained("bert-large-uncased-whole-word-masking").to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking").to(self.device)
        self.toxic_words = set()
        with open(toxic_words_path, 'r') as f:
            for line in f:
                self.toxic_words.add(line.strip())
        self.bad_tokens = self.tokenizer.convert_tokens_to_ids(set(self.tokenizer.tokenize(" ".join(self.toxic_words))))

    def __call__(self, text):
        tokens = self.tokenizer.tokenize(text)
        detox_tokens = []

        for token in tokens:
            if token in self.toxic_words:
                masked_text = tokens[:]
                masked_text[masked_text.index(token)] = '[MASK]'

                # Convert the masked tokens back to a string
                masked_text = ' '.join(masked_text)

                # Masked input should have [CLS] at the beginning and [SEP] at the end
                masked_text = '[CLS] ' + masked_text + ' [SEP]'

                # Tokenize the masked text
                tokenized_text = self.tokenizer.tokenize(masked_text)

                # Convert the tokenized text to input features
                input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

                # Get predictions for the masked tokens
                with torch.no_grad():
                    out = self.model(input_ids)
                    pred = out.logits

                # Find the masked token's index
                mask_idx = tokenized_text.index('[MASK]')

                # Get the predicted probabilities for the [MASK] token
                pred_prob = pred[0, mask_idx]

                # Get the prediction, if it is not a bad word
                while True:
                    pred_id = torch.argmax(pred_prob).item()
                    if pred_id in self.bad_tokens:
                        pred_prob[pred_id] = -float("inf")
                    else:
                        break

                predicted_token = self.tokenizer.convert_ids_to_tokens([pred_id])[0]

                # Replace the [MASK] token with the predicted token
                detox_tokens.append(predicted_token)
            else:
                detox_tokens.append(token)
        result = ' '.join(detox_tokens).replace(' ##', '')
        return result
