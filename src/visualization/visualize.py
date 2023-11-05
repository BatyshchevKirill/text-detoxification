import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def visualize_toxicity(df: pd.DataFrame):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].hist(df.tox_lvl, bins=100)
    axs[0].set_title('Toxicity levels of toxic texst')

    axs[1].hist(df.dtx_lvl, bins=100)
    axs[1].set_title('Toxicity levels of detoxified texsts')

    axs[2].hist(df.tox_diff, bins=100)
    axs[2].set_title('Difference in toxicity levels')

    plt.tight_layout()

    plt.show()


def visualize_toxic_lengths(df: pd.DataFrame):
    plt.hist(df["toxic"].str.len(), bins=1000)
    plt.title("Lengths of toxic texts")
    plt.show()


def plot_similarity(df: pd.DataFrame):
    plt.hist(df.similarity, bins=1000)
    plt.show()


def plot_corr_matrix(df: pd.DataFrame):
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def plot_loss(path, step):

    # Create an event accumulator
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    # Get the training and validation loss data
    train_loss = event_acc.Scalars('Training loss')
    val_loss = event_acc.Scalars('Val loss')

    # Extract the steps and values for training and validation loss
    train_steps, train_values = zip(*[(int(s.step), s.value) for s in train_loss])
    train_steps_filtered = [train_steps[i] for i in range(len(train_steps)) if i % step == 0]
    train_values_filtered = [train_values[i] for i in range(len(train_values)) if i % step == 0]
    val_steps, val_values = zip(*[(int(s.step), s.value) for s in val_loss])

    # Create loss plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(train_steps_filtered, train_values_filtered, label='Training Loss', color='blue')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Training and Validation Loss')
    axs[0].grid(True)

    axs[1].plot(val_steps, val_values, label='Validation Loss', color='red')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Summed Loss')
    axs[1].legend()
    axs[1].set_title('Training and Validation Loss')
    axs[1].grid(True)
    axs[1].show()

