#!/usr/bin/python

import collections
import json
import re
import sys
from json import JSONDecoder, JSONDecodeError


# Decode method for stacked json
# Usage: data = decode_stacked(input_file.read().replace('\n', ''))
def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = re.compile(r'[^\s]').search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            raise Exception("Decoder exception: can not decode stacked json input")
        yield obj


# Read input using json module
for row in sys.stdin:

    # Read line into json format
    raw = json.loads(row)

    # Remove any weird characters from the first column, i.e. the text column
    # Lastly, find words including apostrophe
    sub_reddit = raw['subreddit']
    line = re.sub(r'^\W+|\W+$', '', raw['body'])
    words = re.findall(r'[a-zA-Z0-9_]+\'?[a-zA-Z0-9_]{0,2}', line)

    # Aggregate the word array into a word dictionary with their counts
    words_dict = collections.defaultdict(lambda: int(0))
    for word in words:
        words_dict[word] = words_dict[word] + 1

    # Print out tf, df and n for mapreduce
    for aggregate_word in words_dict:
        print(sub_reddit + " " + aggregate_word.lower() + " tf\t" + str(words_dict[aggregate_word]))
        print(sub_reddit + " " + aggregate_word.lower() + " df\t1")
        print(sub_reddit + "\t1")
