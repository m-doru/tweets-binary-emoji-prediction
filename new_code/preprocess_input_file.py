import os

FILES_AND_LABELS_FULL = {"train_pos_full.txt" : '1', 'train_neg_full.txt' : '-1'}
FILES_AND_LABELS_PARTIAL = {"train_pos.txt" : '1', 'train_neg.txt' : '-1'}
INPUT_TEST_FILE = os.path.join('..', "data", 'twitter-datasets', 'test_data.txt')

OUTPUT_TRAIN_FILE_FULL = 'train_total_full.txt'
OUTPUT_TRAIN_FILE_PARTIAL = 'train_total_partial.txt'
OUTPUT_TEST_FILE = 'test_total.txt'
LABEL_IDENTIFIER = '__label__'
SEPARATOR = ' , '

with open(os.path.join('processed_data', OUTPUT_TRAIN_FILE_PARTIAL), 'w') as f:
    for filename, label in FILES_AND_LABELS_PARTIAL.items():

        with open(os.path.join('..', "data", 'twitter-datasets', filename), 'r') as g:
            for line in g:
                f.write(LABEL_IDENTIFIER + label + SEPARATOR + line.strip() + '\n')

with open(os.path.join('processed_data', OUTPUT_TRAIN_FILE_FULL), 'w') as f:
    for filename, label in FILES_AND_LABELS_FULL.items():

        with open(os.path.join('..', "data", 'twitter-datasets', filename), 'r') as g:
            for line in g:
                f.write(LABEL_IDENTIFIER + label + SEPARATOR + line.strip() + '\n')


with open(os.path.join('processed_data', OUTPUT_TEST_FILE), 'w') as f:
    with open(INPUT_TEST_FILE, 'r') as g:
        for line in g:
            f.write(line.strip().split(',')[1] + '\n')




