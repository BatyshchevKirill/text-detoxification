import argparse
import os
import sys

# Add the current directory to the system path
sys.path.append(os.getcwd())

import numpy as np
from sentence_transformers import SentenceTransformer
from torch.nn import Softmax
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.models.preprocess import file_path


def bleu(input_data: (str, str)) -> float:
    """
    Calculate BLEU score for a pair of reference and translation sentences.

    :param input_data: A tuple containing reference and translation sentences.
    :return float: The calculated BLEU score.
    """
    ref, trans = input_data
    score = BLEUScore(smooth=True)
    score = score([trans], [[ref]]).item()
    return score


def rouge(input_data: (str, str)) -> (float, float):
    """
    Calculate ROUGE scores for a pair of reference and translation sentences.

    :param input_data: A tuple containing reference and translation sentences.
    :return tuple: A tuple containing ROUGE-1 and ROUGE-2 F1 scores.
    """
    ref, trans = input_data
    score = ROUGEScore(rouge_keys=("rouge1", "rouge2"))(trans, ref)
    rouge1_f1 = score["rouge1_fmeasure"]
    rouge2_f1 = score["rouge2_fmeasure"]
    return rouge1_f1, rouge2_f1


class ToxicityClassifier:
    """
    The pretrained model for determining the toxicity of texts
    """
    def __init__(self):
        """
        Initialize the Toxicity Classifier.

        This class uses the RobertaForSequenceClassification model for toxicity classification.
        """
        self.model = RobertaForSequenceClassification.from_pretrained(
            "SkolkovoInstitute/roberta_toxicity_classifier"
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(
            "SkolkovoInstitute/roberta_toxicity_classifier"
        )
        self.sm = Softmax(dim=0)

    def __call__(self, text: str) -> float:
        """
        Classify the toxicity of a given text.

        :param text: The input text to classify.
        :return float: The toxicity score for the input text.
        """
        batch = self.tokenizer.encode(text, return_tensors="pt")
        res = self.model(batch)
        res = self.sm(res.logits.squeeze())
        return res[1]


class SemanticSimilarityClassifier:
    """
    The pretrained model wrapper to calculate the semantic similarity of two sentences
    """

    def __init__(self):
        """
        Initialize the Semantic Similarity Classifier.

        This class uses the SentenceTransformer model for semantic similarity classification.
        """
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, sentences: (str, str)) -> float:
        """
        Calculate the semantic similarity between two sentences.

        :param sentences: A tuple of two sentences for similarity comparison.
        :return float: The calculated semantic similarity score.
        """
        assert len(sentences) == 2, "There should be only 2 sentences for the metric"
        embeddings = self.model.encode(sentences)
        return float(np.dot(*embeddings))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate model parser")
    parser.add_argument("predictions_path", type=file_path)
    parser.add_argument(
        "metric_name", choices=["bleu", "rouge", "toxicity", "similarity"]
    )
    parser.add_argument("-r", "--reference-path", default=None, type=file_path)
    args = parser.parse_args()
    m_name = args.metric_name

    # Read the data, choose the metric
    if m_name == "bleu" or m_name == "rouge" or m_name == "similarity":
        if args.reference_path is None:
            parser.error(
                f"Reference path is required for {m_name} score. Use -r and fill the path of the reference file"
            )

        with open(args.predictions_path, "r") as f:
            predictions = f.read().split("\n")
        with open(args.reference_path, "r") as f:
            reference = f.read().split("\n")
        assert len(predictions) == len(
            reference
        ), "The lengths of the reference and prediction files should be the same"
        data = list(zip(predictions, reference))

        if m_name == "bleu":
            metric = bleu
        elif m_name == "rouge":
            metric = rouge
        elif m_name == "similarity":
            metric = SemanticSimilarityClassifier()
    else:
        with open(args.predictions_path, "r") as f:
            data = f.read().split("\n")
        metric = ToxicityClassifier()

    # Compute the average metric over all samples
    if m_name != "rouge":
        results = 0

        for item in tqdm(data):
            results += metric(item)

        print(
            f"Average {m_name} metric for the given texts is: {results / len(data):.4f}"
        )
    else:
        res1, res2 = 0, 0
        for item in tqdm(data):
            res = metric(item)
            res1 += res[0]
            res2 += res[1]
        print(
            f"Average rouge1 f1 metric for the given texts is: {res1 / len(data):.4f}"
        )
        print(
            f"Average rouge2 f1 metric for the given texts is: {res2 / len(data):.4f}"
        )
