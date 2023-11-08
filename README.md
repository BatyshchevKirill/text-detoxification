# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

## Student Info

**Full Name**: Kirill Batyshchev  
**Email Address**: k.batyshchev@innopolis.university  
**Group Number**: B21-DS-02

## Installation Steps

The following instructions are to be executed using **Python v3.10**.

#### 1. Clone the Repository

```console
git clone https://github.com/BatyshchevKirill/text-detoxification.git
```

#### 2. Change the directory to the project root
```console
cd text-detoxification
```
#### 3. Install Required Python Libraries

```console
pip install -r requirements.txt
```

## Data Processing
### Dataset processing

If you want to use your own data, do the following:

1. Prepare your `*.tsv` file or zipped `*.tsv` file.
2. It should contain the same fields as the initial paraMNT dataset
3. Call the script to prepare the dataset.

```bash
python src/models/preprocess.py <path to source file .tsv or .zip> <path to result file .csv> [-u <unzipped .tsv save path>]
```

## Model Training

To train a model, choose a `model_name` from the list - ["transformer"], specify the `save_path` for the model file (*.pth), and provide the `dataset_path` for the *.csv file. 

To train a model, execute the training script.

```bash
python src/models/train_model.py [transformer] *.SAVE_PATH *.DATASET --vocab-path VOCAB_FILE.pth --batch_size BATCH_SIZE --random-state RANDOM_STATE --epochs EPOCH_NUM
```

For example:

```bash
python src/models/train_model.py transformer src/models/transformer.pth data/interim/preprocessed_data.csv --vocab-path data/interim/vocab.pth --epochs 10
```

The trained model will be saved in `src/models/transformer.pth`.

## Model loading
If you want to use one of the pretrained models (currently supported only one model), you may use the following script:
```console
python src/models/checkpoint_download.py [transformer]
```
The weights for the chosen model will be downloaded to models/{model_name}_checkpoints/{model_name}.pth
## Prediction

If you have a trained model or a pre-trained model and you want to make predictions based on some data, create a *.txt file and input the text there. To generate predictions for your file, use the following command format:

```bash
python src/models/predict_model.py [transformer|t5|baseline] SOURCE_FILE RESULT_FILE_PATH -v VOCAB_FILE_NAME.pth -t TOXIC_WORDS_FILE_NAME.txt -c CHECKPOINT_FILE_NAME.pth
```
The parameters -v (--vocab-path) and -c (--checkpoint) are mandatory
for the transformer model. The parameter -t (--toxic_words_path) is mandatory for running
a baseline model. The results will be saved to the file stated in SAVE_PATH<br>

Examples:

* For T5 model:

```bash
python src/models/predict_model.py t5 data/interim/t5 data/interim/t5_res
```

* For Baseline model:

```bash
python src/models/predict_model.py baseline data/interim/baseline data/interim/baseline_res --toxic_words_path data/interim/toxic_words.txt
```
## Result Testing
You can test the results of translation using BLEU or ROUGE F1 scores,
or evaluate toxicity or semantic similarity of the texts and translation.
Use the following script:
```bash
python src/models/evaluate.py PREDICTIONS_PATH [bleu|rouge|toxicity|similarity] -r REFERENCE_PATH
```
For this script, the prediction and reference files are .txt files with new-line separated
sentences. The reference file is required for BLEU, ROUGE, and semantic similarity metrics.

Example:
```bash
python src/models/evaluate.py data/interim/testing/baseline_results.txt bleu -r data/interim/testing/target_test.txt
```
