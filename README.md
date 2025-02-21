# code-n-gram
Assignment 1 CS520

Extracts methods from a Java repository and trains an n-gram model on them.

## Steps to run:

Prerequisites:
1. Clone the repository
2. Unzip compresseddata.csv.gz to "data.csv"

Combined:
3. run main.py which should extract methods, preprocess them, train the n-gram model, and evaluate it
python ./src/main.py

Separate:
3. Run mining.py to extract methods from the repository which should save them to "methods.csv"
python ./src/data_processing/mining_python_methods.py

4. run preprocessing_java_methods.py to preprocess the methods which should save them to "preprocessed_methods.csv"
python ./src/data_processing/preprocessing_java_methods.py

5. run training.py to train and evaluate the n-gram model
python ./src/model/training.py --data ./data/processed_methods.csv --output_dir ./results --n 7


## Results:

Results should be saved to ./results/
