import argparse
from classifier import *

parser = argparse.ArgumentParser()

parser.add_argument('--pospath', action='store', dest='pos_path', required=True)
parser.add_argument('--negpath', action='store', dest='neg_path', required=True)

args = parser.parse_args()

run(args.pos_path, args.neg_path)
