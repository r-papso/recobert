# RecoBERT

This repository contains our implementation of the paper RecoBERT: A Catalog Language Model for Text-Based Recommendations.

The paper can be found here: https://arxiv.org/pdf/2009.13292.pdf

## Requirements

```bash
python==3.9.15
pandas==1.5.2
pytorch==1.13.0
transformers==4.25.1
```

## Datasets

Preprocessed wines dataset used for RecoBERT training and evaluation along with annotated test subset can be found on this [link](https://drive.google.com/file/d/1oxtPgy14t3rdd5g_AVpiRRfquZU-5MFJ/view?usp=sharing).

## Usage

To train and/or evaluate the RecoBERT implementation:

1. Download preprocessed wines dataset from this [link](https://drive.google.com/file/d/1oxtPgy14t3rdd5g_AVpiRRfquZU-5MFJ/view?usp=sharing).
2. Save the dataset to the ./data folder within this repository.
3. Run training by using command: ```python ./scripts/training.py wines```.
4. After training, run inference by using command: ```python ./scripts/inference.py wines```.
