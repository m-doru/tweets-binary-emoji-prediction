import requests
import string
from bs4 import BeautifulSoup
import os

URL = 'https://www.noslang.com/dictionary/'
save_to_filepath = "dict_noslang"

bad_words_dict = {}
with open('bad_words', 'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    censored, uncensored = line.split(' ')
    bad_words_dict[censored] = uncensored

def uncensor(text):
  '''
  This function takes a string and using the bad_words_dict replaces all the censored words in the string with their
  uncensored counterpart
  :param text:
  :return: text with uncensored words
  '''
  for censored in sorted(bad_words_dict.keys(), key=lambda item: (-len(item), item)):
    text = text.replace(censored, bad_words_dict[censored])
  return text

def parse_noslang():
  '''
  This function parses noslang.com/dictionary/ and creates a dictionary mapping abbreviated/slang words to their
  proper translation. It also saves the dictionary in the filepath specified by save_to_filepath
  :return: dictionary from slang to common words
  '''

  abreviations = []
  words = []

  for letter in string.ascii_lowercase + '1':
    print('Parsing url:{}'.format(URL+letter))
    r = requests.get(URL+letter)
    page_body = r.text

    soup = BeautifulSoup(page_body, 'html.parser')

    abreviations_tags = soup.find_all('dt')
    for abv in abreviations_tags:
      txt = abv.get_text()[:-2]
      abreviations.append(txt)

    words_tags = soup.find_all('dd')
    for tag in words_tags:
      if tag is not None:
        text = tag.get_text()
        res = uncensor(text)
        words.append(res)

  abv_dict = dict(zip(abreviations, words))

  with open(save_to_filepath, 'w') as f:
    for k, v in abv_dict.items():
      if k is not None and v is not None:
        f.write(k + ' ' + v + '\n')
      else:
        print(k, v)

  return abv_dict

def create_no_slang_dict():
  '''
  This function creates a python dictionary mapping slang to common words. It looks for the file containing these
  mappings, otherwise it will parse the website
  :return: dictionary from slang to common words
  '''
  abv_dict = {}

  if os.path.isfile(save_to_filepath):
    with open(save_to_filepath, 'r') as f:
      for line in f.readlines():
        line = line.rstrip()
        try:
          abv, word = line.split(' ', 1)
        except:
          pass
        if abv is not None and word is not None:
          abv_dict[abv] = word
  else:
    abv_dict = parse_noslang()

  return abv_dict

if __name__ == '__main__':
  create_no_slang_dict()
