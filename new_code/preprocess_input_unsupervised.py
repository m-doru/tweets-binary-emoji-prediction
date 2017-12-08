import os

TRAIN_FILES = ['train_pos_full.txt', 'train_neg_full.txt']
TEST_FILE = 'test_data.txt'

OUTPUT_FILE = os.path.join('processed_data', 'unsupervised_total.txt')

with open(OUTPUT_FILE, 'w') as f:
    for FILENAME in TRAIN_FILES:
        with open(os.path.join('..', "data", 'twitter-datasets', FILENAME), 'r') as g:
            f.writelines(g.readlines())

    with open(os.path.join('..', "data", 'twitter-datasets', TEST_FILE), 'r') as g:
        for line in g:
            tweet = line.split(',', 1)[1]
            f.write(tweet + '\n')
