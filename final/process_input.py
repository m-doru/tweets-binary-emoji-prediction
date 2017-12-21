import codecs
import os

FILES_FULL = ["train_pos_full.txt", 'train_neg_full.txt']
FILES_PARTIAL = ["train_pos.txt", 'train_neg.txt']


def process(process_function, file_append):
    for filename in FILES_PARTIAL + FILES_FULL:
        with codecs.open(os.path.join('..', "data", 'twitter-datasets', filename), 'r', encoding='utf8') as g, \
                codecs.open(os.path.join('processed_data', file_append + '_' + filename), 'w', encoding='utf8') as f:

            tweets = g.readlines()
            processed_tweets = process_function(tweets)

            for tweet in processed_tweets:
                f.write(tweet + '\n')

        print("Processed file: " + filename)


def process_remove_duplicates(tweets):
    tweets = list(map(lambda x: x.strip(), tweets))
    return list(set(tweets))


process(process_remove_duplicates, 'no_dups')
