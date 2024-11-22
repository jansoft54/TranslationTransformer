# Vanilla Transformer Translation Model

This repository contains an implementation of a vanilla Transformer-based translation model, designed for translating sequences from a source language to a target language. The implementation adheres closely to the original Transformer architecture described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Features

- **Vanilla Transformer Architecture**: Implements the encoder-decoder structure with multi-head attention and feed-forward layers.
- **Tokenization**: Prepares input and target sequences using token embeddings and learnable positional encodings.
- **Masking**: Handles padding and causal masks to ensure proper attention mechanism behavior.
- **Dataset**: Tested on the WMT14 English-German translation dataset.

## How It Works

1. **Data Preprocessing**:
   - Tokenizes input and target sequences.
   - Adds padding masks and shifts labels for training.

2. **Model Architecture**:
   - Encoder processes the source sequence to generate contextual representations.
   - Decoder uses these representations, applying self-attention and cross-attention to generate the target sequence.

3. **Training**:
   - Outputs are reshaped for compatibility with the loss function.
   - CrossEntropyLoss computes the error, and the model is optimized to minimize it.

## Next Steps

This implementation serves as a foundational model for translation tasks. Future work will extend this architecture to state-of-the-art large language models, such as Meta's Llama 3.

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)

## Usage

Clone the repository, install the dependencies, and run the model on your desired dataset. Check the source code for detailed comments and explanations.
