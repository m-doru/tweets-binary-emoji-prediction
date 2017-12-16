import codecs
import itertools
import os
import re
from multiprocessing import Pool

FILES_FULL = ["train_pos_full.txt", 'train_neg_full.txt']
FILES_PARTIAL = ["train_pos.txt", 'train_neg.txt']
ENGLISH_DICTIONARY_FILE = os.path.join('dicos', 'english_dictionary.txt')

compiled_regexes = {re.compile(r"\'ve"): " have",
                    re.compile(r"\'re"): " are",
                    re.compile(r"\'d"): " would",
                    re.compile(r"\'ll"): " will",
                    re.compile(r"\s{2,}"): ' '}


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


def load_english_dictionary():
    en_dict = set()
    with open(ENGLISH_DICTIONARY_FILE, 'r') as f:
        for line in f:
            en_dict.add(line.strip())

    return en_dict


def remove_repetitions(tweet, dictionary):
    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)

    """
    processed_tweet = ""

    for word in tweet.split():
        word = ''.join(''.join(s)[:2] for _, s in itertools.groupby(word)).replace('#', '')
        if len(word) > 0 and (not (word in dictionary)):
            word = ''.join(''.join(s)[:1] for _, s in itertools.groupby(word))
        processed_tweet += (word + ' ')
    return processed_tweet[:-1]


def correct_spell(tweet, slang_dictionary):
    """
    Function that uses the three dictionaries that we described above and replace noisy words

    Arguments: tweet (the tweet)

    """
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in slang_dictionary.keys():
            tweet[i] = slang_dictionary[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet


def clean(tweet, slang_dictionary, dictionary):
    """
    Function that cleans the tweet using the functions above and some regular expressions
    to reduce the noise

    Arguments: tweet (the tweet)

    """

    for regex, replace_with in compiled_regexes.items():
        tweet = regex.sub(replace_with, tweet)
    tweet = correct_spell(tweet, slang_dictionary)
    tweet = remove_repetitions(tweet, dictionary)

    return tweet


def construct_dictionaries():
    dico = {}
    dico1 = open('dicos/dico1.txt', 'rb')
    for word in dico1:
        word = word.decode('utf8')
        word = word.split()
        dico[word[1]] = word[3]
    dico1.close()
    dico2 = open('dicos/dico2.txt', 'rb')
    for word in dico2:
        word = word.decode('utf8')
        word = word.split()
        dico[word[0]] = word[1]
    dico2.close()
    dico3 = open('dicos/dico2.txt', 'rb')
    for word in dico3:
        word = word.decode('utf8')
        word = word.split()
        dico[word[0]] = word[1]
    dico3.close()

    d = load_english_dictionary()

    return dico, d

#slang_dictionary, dictionary = construct_dictionaries()

def clean_tweet(tweet):
    return clean(tweet, slang_dictionary, dictionary)

def process_last_year(tweets):
    """
    Below, we used three normalizazion dictionaries from those links :
    http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
    http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
    http://luululu.com/tweet/typo-corpus-r1.txt

    These dictionaries have been built by researchers from the noisy tweets and corrects
    the common mistakes and abbreviations that are made in the english
    They help us clean the noise.

    """

    with Pool(processes=8) as pool:
        tweets_processed = pool.map(clean_tweet, tweets)

    return process_remove_duplicates(tweets_processed)


process(process_remove_duplicates, 'no_dups')
