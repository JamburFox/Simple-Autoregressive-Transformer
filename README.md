# Simple-Autoregressive-Transformer
A simple autoregressive decoder only transformer model using PyTorch in Python


This project is a simple implementation of an autoregressive decoder-only transformer model for generative text tasks.

# Dataset
```
+-- dataset\
    +-- convert_to_h5py.py
    +-- corpus.txt
    +-- dataset.hdf5
```
You can build raw text into a hdf5 dataset by running: `convert_to_h5py.py`.
### Parameters
- **--text_file_path**: The location of the raw text dataset (default = "./corpus.txt")
- **--save_path**: The location to save the dataset file to (default = "./dataset.hdf5")
- **--window_size**: The sliding window size (default = 512)
- **--hop_size**: The window hop size (default = 256)

This uses a sliding window approach over the entire text file to build the dataset.

# Tokenizer
```
+-- models\
    +-- tokenizer\
        +-- train_sentencepiece.py
        +-- corpus.txt
        +-- tokenizer.model
```
The tokenizer uses sentencepiece which can be tuned to a custom dataset by running: `train_sentencepiece.py`.
### Parameters
- **--input**: The location of the input text file (default = "./corpus.txt")
- **--output**: The location to save the tokenizer model to (default = "./tokenizer")
- **--vocab_size**: The vocab size (default = 20000)

> **Note**: Special tokens "\<sep>", "\<user>" and "\<bot>" are also added to the tokenizer in the script for demonstration purposes.

> **Note**: The tokenizer should use a seperate local corpus.txt file instead of the full corpus.txt file used in /dataset due to performance reasons.

# Train
```
+-- train.py
```
You can train the model by running: `train.py`.
### Parameters
- **--dataset**: The location of hdf5 dataset (default = "./dataset/dataset.hdf5")
- **--epochs**: Number of epochs to tain for (default = 10)
- **--batch_size**: Batch size of each step (default = 16)
- **--learning_rate**: Optimizer learning rate (default = 1e-3)
- **--scheduler_step**: Number of batches before adjusting learning rate (default = 5)
- **--scheduler_gamma**: Learning rate multiplier for each scheduler step (default = 0.5)
- **--device**: The device to use (default = automatically decided based on system)

This script will try and load the model from `./models/transformer_model.pt` to continue training and will also save the model every epoch to the same path.

# Run
```
+-- run.py
```
You can run the model by running: `run.py`.
### Parameters
- **--text_input**: The input text list to run the model with (default = "hello world")
- **--device**: The device to use (default = automatically decided based on system)

This script will try and load the model from `./models/transformer_model.pt` to run.