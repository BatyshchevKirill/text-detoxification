import argparse
from tqdm import tqdm
from src.data.preprocess import file_path, read, lower, expand_contractions
from src.models import baseline as b, pretrained_t5 as t5, transformer as tr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict model parser")
    parser.add_argument("model_name", choices=["transformer", "t5", "baseline"])
    parser.add_argument("data_path", type=file_path)
    parser.add_argument("save_path", type=file_path)
    parser.add_argument("--vocab-path", default=None, type=file_path)
    parser.add_argument("--toxic_words_path", default=None, type=file_path)
    parser.add_argument("--checkpoint", type=file_path, default=None)
    args = parser.parse_args()

    data = read(args.data_path, train=False)
    data = lower(data)
    data = expand_contractions(data)

    res = []
    if args.model_name == 'transformer':
        pass
    else:
        if args.model_name == 't5':
            model = t5.PretrainedT5()
        else:
            model = b.BaselineModel(args.toxic_words_path)
        for text in tqdm(data.values):
            res.append(model(text))

    res = '\n'.join(res)
    with open(args.save_path, 'w') as f:
        f.write(res)

    print("Results saved to", args.save_path)




