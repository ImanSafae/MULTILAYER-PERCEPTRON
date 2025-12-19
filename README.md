# MULTILAYER PERCEPTRON
This program trains a multi-layer perceptron to diagnose healthy or sick cells into a 'M' or 'B' diagnosis (binary classification).

## Split into training and test sets
Split the provided data into training and test datasets with:
```
python3 evaluation.py
```
The evaluation.py program has been provided by the 42 Network.

## Split the training set into actual training and validation sets
Then split the training dataset into actual training and validation datasets with:
```
python3 separate_dataset.py data_training.csv
```
This will separate your data with a 0.8/0.2 ratio.

## Preprocess data
The data needs to be preprocessed first: the numerical values with be standardized and the alphabetical values will be extracted as the diagnosis and dropped. This assumes that the second column will contain the diagnosis.
The first column will be assumed to contain the IDs and dropped. The NaNs will also be dropped.
```
python3 preprocess.py training_set.csv
python3 preprocess.py validation_set.csv
python3 preprocess.py data_test.csv
```

## Training phase
Finally, you can train your model. You can choose the amount of hidden layers with --layers followed by the number you wish. Otherwise, it will be set as 2 by default.
```
python3 train.py --train preprocessed_training_set.csv --valid preprocessed_validation_set.csv
```

## Prediction phase
At last, predict with:
```
python3 predict.py --dataset preprocessed_data_test.csv --model model/
```