# Named Entity Recognition in the Historical Meeting Protocols of the Tartu City Council (Estonian)

This repository contains code and data for named entity recognition (NER) experiments on historical Estonian texts. The project covers data preprocessing, conversion into CoNLL format, dataset splitting, hyperparameter tuning, model training, and evaluation.

## Data
These materials originate from The National Archives of Estonia, and have been manually annotated with named entities in the project **"Information extraction by the example of protocols of historical institutions (1880–1940)" (EKKD-TA10)**.
The project EKKD-TA10 is funded by the National Program **"Estonian Language and Culture in the Digital Age"**.

### Original data

The original data consists of JSON files containing manually annotated texts. These raw JSON files are not notebooks and represent the primary source data.

The raw data is stored in:
```bash
data/raw/
```
### Processed data (CoNLL format)

The original JSON data is converted into **CoNLL-style BIO format** (one token per line, tab-separated columns, empty line marking sentence boundaries).

The processed datasets are provided in:

```bash
data/conll/
├── train.conll
├── dev.conll
└── test.conll
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks used for:

- data inspection and preprocessing
- conversion from JSON to BIO/CoNLL format
- exploratory experiments
- model training and evaluation


The main processing and training pipelines are implemented as standalone Python scripts in the `scripts/`  directory to ensure reproducibility.


## Data processing pipeline

1. Preprocessing and filtering of raw JSON data
2. Conversion of JSON data into BIO / CoNLL format
3. Splitting the data into train/dev/test sets
4. Hyperparameter tuning
5. Model training
6. Model evaluation

Each step is implemented as a separate script in the `scripts/` directory.

## The best model

[est-roberta-hist-ner-for-tccp](https://huggingface.co/tartuNLP/est-roberta-hist-ner-for-tccp)
