# Text Cleansing Assignment

## Student Details

**Full Name**: Kirill Batyshchev  
**Email Address**: k.batyshchev@innopolis.university  
**Group Number**: B21-DS-02

## Installation Steps

The following instructions are to be executed using **Python v3.10**.

#### 1. Clone the Repository

```console
git clone https://github.com/BatyshchevKirill/text-detoxification.git
```

#### 2. Install Required Python Libraries

```console
pip install -r text-detoxification/requirements.txt
```

## Data Processing
### Custom Data Preprocessing

If you prefer to use your own data, follow the instructions below:

1. Prepare your `*.tsv` file or zipped `*.tsv` file.
2. Call the script to prepare the data and create the vocabulary.

```bash
python /src/models/preprocess.py <path to source file .tsv or .zip> <path to result file .csv> [-u <unzipped .tsv file path>]
```

## Model Training

To train a model, choose a `model_name` from the list - ["transformer"], specify the `save_path` for the model file (*.pth), and provide the `dataset_path` for the *.csv file. 

To train a model, execute the training script.

```bash
python text-detoxification/src/models/train_model.py [transformer] *.SAVE_PATH *.DATASET --vocab-path VOCAB_FILE.pth --batch_size BATCH_SIZE --random-state RANDOM_STATE --epochs EPOCH_NUM
```

For example:

```bash
python text-detoxification/src/models/train_model.py transformer text-detoxification/src/models/transformer.pth text-detoxification/data/interim/preprocessed_data.csv --vocab-path text-detoxification/data/interim/vocab.pth --epochs 10
```

The trained model will be saved in `text-detoxification/src/models/transformer.pth`.

## Prediction

If you have a trained model or a pre-trained model and you want to make predictions based on some data, create a *.txt file and input the text there. To generate predictions for your file, use the following command format:

```bash
python text-detoxification/src/models/predict_model.py [transformer|t5|baseline] SOURCE_FILE RESULT_FILE_PATH --vocab-path VOCAB_FILE_NAME.pth --TOXIC_WORDS_FILE_NAME.txt --checkpoint CHECKPOINT_FILE_NAME.pth
```

Examples:

* For T5 model:

```bash
python text-detoxification/src/models/predict_model.py t5 text-detoxification/data/interim/t5 data/interim/t5_res
```

* For Baseline model:

```bash
python src/models/predict_model.py baseline text-detoxification/data/interim/baseline text-detoxification/data/interim/baseline_res --toxic_words_path text-detoxification/data/interim/toxic_words.txt
```