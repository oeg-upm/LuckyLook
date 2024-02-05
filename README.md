# LuckyLook

LuckyLook is a tool designed for recommending scientific journals for researchers looking to publish their work. It uses a Transformer model with additional graph level information
to provide a better recommendation.

## Installation

### Dataset Setup

1. Download the dataset from Zenodo: https://doi.org/10.5281/zenodo.8386011
2. Create a `Dataset/` folder in the LuckyLook directory.
3. Move the downloaded dataset into the `Dataset/` folder.

## Usage

JSON files for configuration are located in the `configs/` folder. These configuration files contain parameters and settings for training and testing the model.

### Training

To train the model, run:

python train.py --config configs/config_BERT_gnn.json

### Testing

For testing, specify the checkpoint directory:

python test.py --resume path/to/checkpoint

## Citation
