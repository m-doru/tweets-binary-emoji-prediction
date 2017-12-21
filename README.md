# Tweets classification - EPFL CS-433

This repo contains code and instructions necessary to classify tweets as containing ':)' or ':('. The corresponding [kaggle competition](https://www.kaggle.com/c/epfml17-text) was part of CS-433 Machine learning class from EPFL.

## Design decisions


## How to run the project and TRAIN the models
### \*nix friendly guide. For other platforms some steps might differ
**Running time** The current model took around 12-hours to train on a 8-core CPU, 60GB of RAM and a Tesla K80 GPU. The GPU is highly recommended.

1. Clone this repo
```
$ git clone https://github.com/m-doru/tweets-sentiment-analysis.git
$ cd tweets-sentiment-analysis
```

2. Install [fastText](https://github.com/facebookresearch/fastText) v0.1.0 with build for Python.
This should be possible after this step:
```
$ python3
>> import fasttext
>>
```

3. Clone [sent2vec](https://github.com/epfml/sent2vec.git) at the root directory of the project. Follow the Setup&Requirments to compile it. Then download the [sent2vec_twitter_bigrams 23GB (700dim, trained on english tweets)](https://drive.google.com/open?id=0B6VhzidiLvjSeHI4cmdQdXpTRHc) [v1](https://github.com/epfml/sent2vec/releases/tag/v1) embeddings and place them in data/ 

4. Download Glove Twitter pretrained word-vectors [glove.twitter.27B.zip](https://nlp.stanford.edu/projects/glove/). Unzip file and place glove.twitter.27B.200d.txt in data/glove/

5. Download the data from the [kaggle competition](https://www.kaggle.com/c/epfml17-text) and place the ```.txt``` files in data/twitter-datasets/.

6. Install the following python requirements:
  * scikit-learn
  * keras with tensorflow backend


### How to run the project to get the pretrained model the kaggle submission 
1. Clone this repo
```
$ git clone https://github.com/m-doru/tweets-sentiment-analysis.git
$ cd tweets-sentiment-analysis
```

2. Download the data from the [kaggle competition](https://www.kaggle.com/c/epfml17-text) and place the ```.txt``` files in data/twitter-datasets/.

3. Install the following python3 requirements:
 * scikit-learn

4. Run run_pretrained.py
