import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from sentence_transformers import SentenceTransformer
from src.models.preprocess import file_path
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import Softmax
from tqdm import tqdm


def bleu(input_data):
    ref, trans = input_data
    score = BLEUScore(smooth=True)
    score = score([trans], [[ref]]).item()
    return score


def rouge(input_data):
    ref, trans = input_data
    score = ROUGEScore(rouge_keys=("rouge1", "rouge2"))(trans, ref)
    rouge1_f1 = score['rouge1_fmeasure']
    rouge2_f1 = score['rouge2_fmeasure']
    return rouge1_f1, rouge2_f1


class ToxicityClassifier:
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained(
            'SkolkovoInstitute/roberta_toxicity_classifier')

        self.tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.sm = Softmax(dim=0)

    def __call__(self, text):
        batch = self.tokenizer.encode(text, return_tensors='pt')
        res = self.model(batch)
        res = self.sm(res.logits.squeeze())
        return res[1]


class SemanticSimilarityClassifier:
    """

    """
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __call__(self, sentences):
        assert len(sentences) == 2, "There should be only 2 sentences for the metric"
        embeddings = self.model.encode(sentences)
        return np.dot(*embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model parser")
    parser.add_argument("predictions_path", type=file_path)
    parser.add_argument("metric_name", choices=["bleu", "rouge", "toxicity", "similarity"])
    parser.add_argument("-r", "--reference-path", default=None, type=file_path)
    args = parser.parse_args()
    m_name = args.metric_name

    if m_name == "bleu" or m_name == "rouge" or m_name == "similarity":
        if args.reference_path is None:
            parser.error(
                f"Reference path is required for {m_name} score. Use -r and fill the path of the reference file"
            )

        with open(args.predictions_path, 'r') as f:
            predictions = f.read().split("\n")
        with open(args.reference_path, 'r') as f:
            reference = f.read().split("\n")
        assert len(predictions) == len(
            reference), "The lengths of the reference and prediction files should be the same"
        data = list(zip(predictions, reference))

        if m_name == 'bleu':
            metric = bleu
        elif m_name == 'rouge':
            metric = rouge
        elif m_name == 'similarity':
            metric = SemanticSimilarityClassifier()
    else:
        with open(args.predictions_path, 'r') as f:
            data = f.read().split("\n")
        metric = ToxicityClassifier()

    if m_name != "rouge":
        results = 0

        for item in tqdm(data):
            results += metric(item)

        print(f"Average {m_name} metric for the given texts is: {results / len(data):.4f}")
    else:
        res1, res2 = 0, 0
        for item in tqdm(data):
            res = metric(item)
            res1 += res[0]
            res2 += res[1]
        print(f"Average rouge1 f1 metric for the given texts is: {res1 / len(data):.4f}")
        print(f"Average rouge2 f1 metric for the given texts is: {res2 / len(data):.4f}")
