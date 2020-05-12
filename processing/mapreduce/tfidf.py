#!/usr/bin/python
import math
import sys
import re

# Variable declarations
document_count = 0
tf = 0
df = 0


# Method for calculating tf-idf
def tfidf(termf, documentf, n):
    t_weight = 0 if termf == 0 else float(1) + math.log(float(termf), 10)
    d_weight = 0 if documentf == 0 else math.log(float(n) / float(documentf), 10)
    return t_weight * d_weight


for line in sys.stdin:
    # Split by tab to get key and value
    # Split the key to obtain subreddit and the word's specs
    key, value = re.split(r'\t', line)
    specs = re.split(r'\s', key)

    # Check to see if within the same subreddit
    # If yes, update the document count and previous subreddit counter
    if len(specs) == 1:
        document_count = value
    else:
        if specs[2] == "df":
            df = value
        elif specs[2] == "tf":
            tf = value
            print(specs[0] + " " + specs[1] + "\t" + str(tfidf(tf, df, document_count)))
