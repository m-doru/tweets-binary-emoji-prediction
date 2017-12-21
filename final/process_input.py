import codecs
import os

FILES_FULL = ["train_pos_full.txt", 'train_neg_full.txt']
FILES_PARTIAL = ["train_pos.txt", 'train_neg.txt']


def process(process_function, file_prefix):
    '''
    Method that processes the input according to a processing function.
    :param process_function: the processing function that should be applied to the the tweets in each file
    :param file_prefix: the prefix of the resulted files
    :return: None
    '''
    for filename in FILES_PARTIAL + FILES_FULL:
        with codecs.open(os.path.join('..', "data", 'twitter-datasets', filename), 'r', encoding='utf8') as g, \
                codecs.open(os.path.join('processed_data', file_prefix + '_' + filename), 'w', encoding='utf8') as f:

            tweets = g.readlines()
            processed_tweets = process_function(tweets)

            for tweet in processed_tweets:
                f.write(tweet + '\n')

        print("Processed file: " + filename)


def process_remove_duplicates(tweets):
    '''
    Processing function for a set of tweets. This function only removes duplicates.
    :param tweets: the tweets to process
    :return: the tweets given as parameter, each one of them present only one time
    '''
    tweets = list(map(lambda x: x.strip(), tweets))
    return list(set(tweets))


process(process_remove_duplicates, 'no_dups')
