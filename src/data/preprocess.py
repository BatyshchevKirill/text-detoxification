import zipfile

import contractions
import pandas as pd


def unpack(content_path: str, target_path: str):
    with zipfile.ZipFile(content_path, "r") as zip_ref:
        zip_ref.extractall(target_path)


def read(content_path: str):
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
    return df


def filter_df(
    df: pd.DataFrame,
    toxic_threshold: float = 0.9,
    detoxed_threshold: float = 0.03,
    max_len: int = 200,
) -> pd.DataFrame:
    df = df[df.tox_lvl > toxic_threshold]
    df = df[df.dtx_lvl < detoxed_threshold]
    df = df[df.toxic.str.len() < max_len]
    return df


def lower(df: pd.DataFrame) -> pd.DataFrame:
    df["toxic"], df["detoxed"] = df["toxic"].str.lower().replace("...", ""), df[
        "detoxed"
    ].str.lower().replace("...", "")
    return df


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["toxic"].str.contains(r"^[a-zA-Z0-9\(\)\s\'.,!?;:-]+$", regex=True)]
    df.reset_index(inplace=True)
    df = df.drop("index", axis=1)
    return df


def expand_contractions(df: pd.DataFrame) -> pd.DataFrame:
    df["detoxed"] = df["detoxed"].apply(contractions.fix)
    df["toxic"] = df["toxic"].apply(contractions.fix)
    df["toxic"] = df["toxic"].str.replace("in'", "ing")
    df["detoxed"] = df["detoxed"].str.replace("in'", "ing")
    return df


def save_preprocessed(df: pd.DataFrame, path: str):
    df = df.drop(["similarity", "length_diff", "tox_lvl", "dtx_lvl"], axis=1)
    df.to_csv(path)
