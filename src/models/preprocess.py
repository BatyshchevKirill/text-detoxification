import argparse as argparse
import os
import zipfile

import contractions
import pandas as pd


def unpack(content_path: str, target_path: str):
    with zipfile.ZipFile(content_path, "r") as zip_ref:
        zip_ref.extractall(target_path)


def read(content_path: str, train=True):
    if not train:
        df = pd.DataFrame()
        with open(content_path) as f:
            lines = f.readlines()
            df["toxic"] = pd.Series(lines)
        return df
    df = pd.read_csv(content_path, sep="\t")
    df.set_index(df.columns[0], inplace=True)
    df.index.name = None
    df["toxic"] = df.apply(
        lambda row: row["reference"]
        if row["ref_tox"] > row["trn_tox"]
        else row["translation"],
        axis=1,
    )
    df["detoxed"] = df.apply(
        lambda row: row["reference"]
        if row["ref_tox"] <= row["trn_tox"]
        else row["translation"],
        axis=1,
    )
    df["tox_lvl"] = df.apply(
        lambda row: row["ref_tox"]
        if row["ref_tox"] > row["trn_tox"]
        else row["trn_tox"],
        axis=1,
    )
    df["dtx_lvl"] = df.apply(
        lambda row: row["ref_tox"]
        if row["ref_tox"] <= row["trn_tox"]
        else row["trn_tox"],
        axis=1,
    )
    df.drop(["reference", "translation", "ref_tox", "trn_tox"], inplace=True, axis=1)
    df = df.rename(columns={'lenght_diff': 'length_diff'})
    return df


def filter_df(
        df: pd.DataFrame,
        toxic_threshold: float = 0.9,
        detoxed_threshold: float = 0.03,
        max_len: int = 200,
        train=True,
) -> pd.DataFrame:
    if not train:
        return df
    df = df[df.tox_lvl > toxic_threshold]
    df = df[df.dtx_lvl < detoxed_threshold]
    df = df[df.toxic.str.len() < max_len]
    return df


def lower(df: pd.DataFrame, train=True) -> pd.DataFrame:
    if train:
        df["detoxed"] = df[
            "detoxed"
        ].str.lower().replace("...", "")
    df["toxic"] = df["toxic"].str.lower().replace("...", "")
    return df


def filter_english(df: pd.DataFrame, train=False) -> pd.DataFrame:
    if not train:
        return df
    df = df[df["toxic"].str.contains(r"^[a-zA-Z0-9\(\)\s\'.,!?;:-]+$", regex=True)]
    df.reset_index(inplace=True)
    df = df.drop("index", axis=1)
    return df


def expand_contractions(df: pd.DataFrame, train=True) -> pd.DataFrame:
    if not train:
        df["detoxed"] = df["detoxed"].apply(contractions.fix)
        df["detoxed"] = df["detoxed"].str.replace("in'", "ing")
    df["toxic"] = df["toxic"].apply(contractions.fix)
    df["toxic"] = df["toxic"].str.replace("in'", "ing")
    return df


def save_preprocessed(df: pd.DataFrame, path: str, train=True):
    if train:
        df = df.drop(["similarity", "length_diff", "tox_lvl", "dtx_lvl"], axis=1)
    df.to_csv(path)


def run_pipeline(df: pd.DataFrame, result_path: str, train=True) -> None:
    filter_df(df, train=train)
    lower(df, train=train)
    filter_english(df, train=train)
    expand_contractions(df, train=train)
    save_preprocessed(df, result_path, train=train)


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess parser")
    parser.add_argument('original_path', type=file_path, help="Required argument for path to original content")
    parser.add_argument('result_path', type=file_path, help="Required argument for path to preprocessed result")
    parser.add_argument('-t', '--train', action='store_true', help='Optional flag for training. Defaults to false')
    parser.add_argument('-u', '--unzip', type=file_path, default=None, help='Paste path to unzip result')
    args = parser.parse_args()
    content_path, result_path, train, unzip = args.original_path, args.result_path, args.train, args.unzip
    if unzip:
        unpack(content_path, unzip)
        content_path = unzip
    df = read(content_path, True) if train else read(content_path)
    run_pipeline(df, result_path, train)