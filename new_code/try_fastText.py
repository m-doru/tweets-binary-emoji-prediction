from fastText import train_supervised
import os

TRAINING_SET = os.path.join('processed_data', 'train_total.txt')
OUTPUT_MODEL = os.path.join('processed_data', 'model')

train_supervised(input=TRAINING_SET, output=OUTPUT_MODEL, lr=0.1, dim=100, ws=5, epoch=5, minCount=2, minCountLabel=0, minn=0, maxn=0, neg=5, wordNgrams=2, loss='softmax', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, verbose=2, saveOutput=1)