# Multilingual Text LLM (From Scratch)

## Overview
This project builds a decoder-style language model from scratch using PyTorch.

## Features
- Custom BPE tokenizer
- Data pipeline (Dataset + DataLoader)
- Training loop
- Text generation (greedy + sampling)

## Run

### Train
python train_full.py

### Generate
python generate.py

## Config
- vocab_size = 5000
- seq_length = 128
- batch_size = 4
