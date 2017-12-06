import os

FILES_AND_LABELS = {"train_pos_full.txt" : '1', 'train_neg_full.txt' : '0'}
INPUT_TEST_FILE = os.path.join('..', "data", 'twitter-datasets', 'test_data.txt')

OUTPUT_TRAIN_FILE = 'train_total.txt'
OUTPUT_TEST_FILE = 'test_total.txt'
LABEL_IDENTIFIER = ' __label__'

with open(os.path.join('processed_data', OUTPUT_TRAIN_FILE), 'w') as f:
    for filename, label in FILES_AND_LABELS.items():

        with open(os.path.join('..', "data", 'twitter-datasets', filename), 'r') as g:
            for line in g:
                f.write(line.strip() + LABEL_IDENTIFIER + label + '\n')

with open(os.path.join('processed_data', OUTPUT_TEST_FILE), 'w') as f:
    with open(INPUT_TEST_FILE, 'r') as g:
        for line in g:
            f.write(line.strip().split(',')[1] + '\n')




