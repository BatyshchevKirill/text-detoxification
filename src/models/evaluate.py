from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import ROUGEScore
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import Softmax


def bleu(ref, trans):
    return bleu_score([trans], [[ref]], max_n=3, weights=[1/3, 1/3, 1/3])


def rouge(ref, trans):
    score = ROUGEScore(rouge_keys=("rouge1", "rouge2"))(trans, ref)
    rouge1_f1 = score['rouge1_fmeasure']
    rouge2_f1 = score['rouge2_fmeasure']
    return rouge1_f1, rouge2_f1


class ToxicityClassifier:
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.sm = Softmax(dim=0)

    def __call__(self, text):
        batch = self.tokenizer.encode(text, return_tensors='pt')
        res = self.model(batch)
        res = self.sm(res.logits.squeeze())
        return res[1]



