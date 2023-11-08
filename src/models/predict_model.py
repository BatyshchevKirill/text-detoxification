import argparse
import os
import sys

sys.path.append(os.getcwd())

from tqdm import tqdm

from src.models import baseline as b
from src.models import pretrained_t5 as t5
from src.models.preprocess import (expand_contractions, file_creatable_path,
                                   file_path, lower, read)
from src.models.transformer_predict import TransformerPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict model parser")
    parser.add_argument("model_name", choices=["transformer", "t5", "baseline"])
    parser.add_argument("data_path", type=file_path)
    parser.add_argument("save_path", type=file_creatable_path)
    parser.add_argument("-v", "--vocab-path", default=None, type=file_path)
    parser.add_argument("-t", "--toxic_words_path", default=None, type=file_path)
    parser.add_argument("-c", "--checkpoint", type=file_path, default=None)
    args = parser.parse_args()

    # Preprocess data
    data = read(args.data_path, train=False)
    data = lower(data, train=False)
    data = expand_contractions(data, train=False)

    res = []

    # Choose the model
    if args.model_name == "t5":
        model = t5.PretrainedT5()
    elif args.model_name == "transformer":
        model = TransformerPredictor((512, 25000, 8, 3, 3, 4, 0.1, 128))
    else:
        model = b.BaselineModel(args.toxic_words_path)

    # Process the texts
    for text in tqdm(data.values):
        res.append(model(text[0]))

    # Save the results
    res = "\n".join(res)
    with open(args.save_path, "w+") as f:
        f.write(res)

    print("Results saved to", args.save_path)
