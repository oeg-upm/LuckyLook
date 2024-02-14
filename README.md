# LuckyLook

LuckyLook is a tool designed for recommending PubMed scientific journals for researchers looking to publish their work. It uses a Transformer model with additional graph-level information
to provide a better recommendation.

## Installation

### Dataset Setup

1. Download the dataset from Zenodo: https://doi.org/10.5281/zenodo.8386011
2. Create a `Dataset/` folder in the LuckyLook directory.
3. Move the downloaded dataset into the `Dataset/` folder.

## Usage

Configuration JSON files, which include parameters and settings for model training, are stored in the `configs/` directory.

### Training

To train the model, run:

```python
python train.py --config configs/<config_file>.json
```

### Testing

For testing, specify the checkpoint directory:

```python
python test.py --resume path/to/checkpoint
```

## Streamlit web application

## Usage

```python
streamlit run LuckyLook.py
```

## Citation
