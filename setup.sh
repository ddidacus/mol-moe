#!/bin/bash

echo "[-] Installing environment..."
conda env create -f environment.yml
conda activate molmoe
pip install gdown
pip install -e .

echo "[-] Setting up directories..."
mkdir -p models
mkdir -p datasets
mkdir -p datasets/raw
mkdir -p support
mkdir -p results

echo "[-] Downloading data..."
gdown 1LU95-zTiQuwoakhp-6Qx-l0YuUYYvz4Q -O support/
gdown 1W1h0XtOeb2qAb7l_KhgdjEt1Wzk-jeW_ -O support/
gdown 1EUgGV4lWehBqq6ZuxM-JU0A9shwTk8Zl -O support/
gdown 1rcJLMzF2VqGq8DgJuyvA-LRJKhkNnRKY -O datasets/raw
