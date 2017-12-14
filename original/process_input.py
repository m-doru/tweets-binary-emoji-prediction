import os
import itertools
import re
import enchant
import codecs

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

def remove_repetitions(tweet, dictionary):

    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)

    """
    tweet = tweet.split()
    for i in range(len(tweet)):
        tweet[i] = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        if len(tweet[i]) > 0:
            if not dictionary.check(tweet[i]):
                tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet = ' '.join(tweet)
    return tweet

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
    # Separates the contractions and the punctuation
    # tweet = re.sub("\'s", " \'s", tweet)
    # tweet = re.sub("\'ve", " \'ve", tweet)
    # tweet = re.sub("n\'t", " n\'t", tweet)
    # tweet = re.sub("\'re", " \'re", tweet)
    # tweet = re.sub("\'d", " \'d", tweet)
    # tweet = re.sub("\'ll", " \'ll", tweet)
    # tweet = re.sub(",", " , ", tweet)
    #
    # tweet = re.sub("\(", " \( ", tweet)
    # tweet = re.sub("\)", " \) ", tweet)
    # tweet = re.sub('\?', " \? ", tweet)
    tweet = str(re.sub("\s{2,}", " ", tweet))
    tweet = remove_repetitions(tweet, dictionary)
    tweet = correct_spell(tweet, slang_dictionary)

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

    d = enchant.Dict('en_US')

    return dico, d

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

    slang_dictionary, dictionary = construct_dictionaries()
    tweets_processed = [clean(tweet, slang_dictionary, dictionary) for tweet in tweets]


    return process_remove_duplicates(tweets_processed)

process(process_last_year, 'last_year')