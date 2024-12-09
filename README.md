# Glaucoma Prediction
## Introduction
This project implements a deep learning framework for automated glaucoma detection from retinal images, combining state-of-the-art computer vision architectures with a production-ready development structure. Glaucoma, being a leading cause of irreversible blindness worldwide, requires early detection for effective treatment. Our framework aims to assist healthcare professionals in screening and diagnosis through automated image analysis.

### Key Objectives
* Clinical Relevance
* Technical Innovation
    * Support easy model switching through a plug-and-play architecture
* Educational Value
    * Demonstrate professional ML project organization
    * Serve as a template for similar medical imaging projects

## Directory Structure
```
root/
├── configs/
│   └── config.yaml          # Training configurations
├── data/
│   ├── raw/                # Original images
│   └── processed/          # Preprocessed datasets
├── models/
│   └── checkpoints/        # Saved model states
├── reports/
│   └── results/            # Evaluation metrics
├── scripts/
│   ├── train.py           # Training entry point
│   ├── evaluate.py        # Evaluation pipeline
│   └── preprocess_data.py # Data preprocessing
└── src/
    ├── data/              # Dataset and preprocessing
    ├── models/            # Model architectures
    ├── training/          # Training logic
    └── evaluation/        # Metrics and analysis
```

## Installation
Strongly recommend one to create virtual environment when developing
```
$ python -m venv venv
```
If you're using Unix base system, use the following command to activate the virtual environment
```
$ source venv/bin/activate
```
For Windows users, please use the alternative
```
$ .\venv\Scripts\activate
```

At the time we can install the dependencies
```
$ pip install -e .
```

## Usage
Please first clone the dataset from [here](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/data).

After that you should place the data under `data/raw/` as the following manner
```
...
├── data/
│   ├── raw/
│       ├── test/
│       ├── train/
│           ├── NRG/
│           ├── RG/
│       ├── validation/
│       └── metadata.csv
│   └── processed/
...
```

### Data Preprocessing
```
$ python scripts/preprocess_data.py
```
### Training
```
$ python scripts/train.py
```
### Evaluation
```
$ python scripts/evaluate.py
```

## Requirements
* Python 3.8+

## License
MIT
