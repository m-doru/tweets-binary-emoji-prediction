import os
import random

FILES_AND_LABELS_FULL = {"train_pos_full.txt" : '1', 'train_neg_full.txt' : '0'}
FILES_AND_LABELS_PARTIAL = {"train_pos.txt" : '1', 'train_neg.txt' : '0'}
INPUT_TEST_FILE = os.path.join('..', "data", 'twitter-datasets', 'test_data.txt')

OUTPUT_TRAIN_FILE_FULL = 'train_total_full.txt'
OUTPUT_TRAIN_FILE_PARTIAL = 'train_total_partial.txt'
OUTPUT_TEST_FILE = 'test_total.txt'
LABEL_IDENTIFIER = ' __label__'

def shuffle_inputs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines_shuffled = random.shuffle(lines)

    with open(filename, 'w') as f:
        f.writelines(lines_shuffled)

with open(os.path.join('processed_data', OUTPUT_TRAIN_FILE_FULL), 'w') as f:
    for filename, label in FILES_AND_LABELS_FULL.items():

        with open(os.path.join('..', "data", 'twitter-datasets', filename), 'r') as g:
            for line in g:
                f.write(line.strip() + LABEL_IDENTIFIER + label + '\n')


with open(os.path.join('processed_data', OUTPUT_TRAIN_FILE_PARTIAL), 'w') as f:
    for filename, label in FILES_AND_LABELS_PARTIAL.items():

        with open(os.path.join('..', "data", 'twitter-datasets', filename), 'r') as g:
            for line in g:
                f.write(line.strip() + LABEL_IDENTIFIER + label + '\n')

shuffle_inputs(os.path.join(os.path.join('processed_data', OUTPUT_TRAIN_FILE_FULL)))
shuffle_inputs(os.path.join(os.path.join('processed_data', OUTPUT_TRAIN_FILE_PARTIAL)))

with open(os.path.join('processed_data', OUTPUT_TEST_FILE), 'w') as f:
    with open(INPUT_TEST_FILE, 'r') as g:
        for line in g:
            f.write(line.strip().split(',')[1] + '\n')




