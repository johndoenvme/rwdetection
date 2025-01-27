# Ransomware Detection using Per-Command Labeled NVMe Streams

## Introduction

This repository is based on the paper "Learning the Language of NVMe Streams for Ransomware Detection" and provides a Python implementation to reconstruct its results and execute other functionalities.

## Installation

To run the code, follow these steps:

1. Install the required modules by running `pip install -r requirements.txt` in your terminal.
2. Ensure you have Python installed on your system.

### Running the Code

To start the application, navigate to the project directory and run `python main.py`. You can choose between two modes:
- **Demo Mode:** Set the `demo` flag to `True` to run a simplified version of the code.
- **Full Execution:** Set the `demo` flag to `False` and download the CLEAR dataset from [Kaggle](https://www.kaggle.com/datasets/johndoenvme/clear-command-level-annotated-ransomware) to the /CLEAR_Dataset folder.

# License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
