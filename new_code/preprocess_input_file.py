import os

FILES_AND_LABELS = {"train_pos.txt" : '1', 'train_neg.txt' : '0'}
TRAIN_FILE = 'train_total.txt'
LABEL_IDENTIFIER = '__label__'

with open(os.path.join('processed_data', TRAIN_FILE), 'w') as f:
    for filename, label in FILES_AND_LABELS.items():

        with open(os.path.join('..', "data", 'twitter-datasets', filename), 'r') as g:
            for line in g:
                f.write(line.strip() + LABEL_IDENTIFIER + label + '\n')



