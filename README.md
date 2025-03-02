# N-gram Model for Java Code Completion

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)  
* [3 Results](#3-results)  

---

# **1. Introduction**  
This project explores **code completion in Java**, leveraging **N-gram language modeling**. The N-gram model predicts the next token in a sequence by learning the probability distributions of token occurrences in training data. The model selects the most probable token based on learned patterns, making it a fundamental technique in natural language processing and software engineering automation.

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/your-repository/code-n-gram.git
```

(2) Navigate into the repository:
```shell
~ $ cd code-n-gram
~/code-n-gram $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:
```shell
~/code-n-gram $ python -m venv ./venv/
~/code-n-gram $ source venv/bin/activate
(venv) ~/code-n-gram $ 
```

For Windows:
```shell
~/code-n-gram $ python -m venv ./venv/
~/code-n-gram $ .\venv\Scripts\activate
(venv) ~/code-n-gram $ 
```

To deactivate the virtual environment, use the command:
```shell
(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:
```shell
(venv) ~/code-n-gram $ pip install -r requirements.txt
```

## **2.3 Run N-gram**

The code provides several ways to run the N-gram model:

### Option 1: Using a processed CSV file

If you have a CSV file with a column named "Method Code No Comments" containing Java methods:
```shell
(venv) ~/code-n-gram $ python src/main.py --processed_file path/to/your/methods.csv --output_dir ./results
```

### Option 2: Using a text file with one method per line

If you have a text file with one Java method per line:
```shell
(venv) ~/code-n-gram $ python src/main.py --training_txt path/to/your/training.txt --output_dir ./results
```

### Additional Parameters

You can customize the model with these parameters:
```shell
--min_n 2             # Minimum n-gram size to try (default: 2)
--max_n 8             # Maximum n-gram size to try (default: 8)
--smoothing_k 0.01    # Smoothing parameter (default: 0.01)
```

Example with custom parameters:
```shell
(venv) ~/code-n-gram $ python src/main.py --processed_file data/processed_methods.csv --min_n 3 --max_n 7 --smoothing_k 0.05
```

# **3. Results**  

The results are saved to the specified output directory (default: `./results/`):

- `metrics.json`: Contains the final model's performance metrics
  - Perplexity
  - Perplexity excluding punctuation
  - Accuracy
  - Vocabulary size

- `metrics_compare.json`: Contains metrics for all n-gram sizes tested during model selection

- `best_model.pkl`: The serialized best model that can be loaded for future use

To interpret the results, look for the perplexity values (lower is better) and accuracy (higher is better) in the metrics files.
