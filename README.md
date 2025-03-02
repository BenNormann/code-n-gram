# N-gram Model for Java Code Completion

* [1. Introduction](#1-introduction)  
* [2. Getting Started](#2-getting-started)  
* [3. Running Options](#3-running-options)
* [4. Results](#4-results)  

## 1. Introduction
This project implements an **N-gram language model** for Java code completion. The model predicts the next token in code by learning probability distributions from training data.

## 2. Getting Started

### Prerequisites
- Python 3.9+
- Required packages: pandas, pygments, scikit-learn, tqdm

### Installation
```bash
# Clone repository
git clone https://github.com/BenNormann/code-n-gram
cd code-n-gram

# Install dependencies
pip install -r requirements.txt
```

## 3. Running Options

### Basic Usage

**Option 1:** Using a CSV file with Java methods
```bash
python src/main.py --processed_file path/to/methods.csv
```

**Option 2:** Using a text file (one method per line)
```bash
python src/main.py --training_txt path/to/training.txt
```

### Quick Mode for Faster Development

For faster experimentation, use the `--quick` flag to train on a smaller sample:

```bash
python src/main.py --processed_file path/to/methods.csv --quick
```

This will:
- Use a random sample of 1000 methods for model selection
- Still train the final model on the full dataset
- Still evaluate on the full test set
- Significantly reduce training time while maintaining final model quality

### Additional Parameters

```bash
--output_dir ./results    # Output directory (default: ./results)
--min_n 2                 # Minimum n-gram size (default: 2)
--max_n 8                 # Maximum n-gram size (default: 8)
--smoothing_k 0.01        # Smoothing parameter (default: 0.01)
```

### Example with Multiple Parameters

```bash
python src/main.py --processed_file data/methods.csv --quick --min_n 3 --max_n 6 --smoothing_k 0.05
```

## 4. Results

Results are saved to the specified output directory:

- `metrics.json`: Final model metrics (perplexity, accuracy, vocabulary size)
- `metrics_compare.json`: Metrics for all tested n-gram sizes
- `best_model.pkl`: Serialized best model

Lower perplexity values and higher accuracy indicate better model performance.
