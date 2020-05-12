#!/usr/bin/python

import sys
import re

previous_key = None
termSum = 0

for line in sys.stdin:
    # Split by tab to get key and value
    key, value = re.split(r'\t', line)

    # Not the same as previous, means new key
    if key != previous_key:
        # If it is not the first element, print and reset
        if previous_key is not None:
            print(previous_key + '\t' + str(termSum))

        previous_key = key
        termSum = 0

    # Else, sum up the values of the same key
    termSum = termSum + int(value)

# Print the last element in the list
print(previous_key + '\t' + str(termSum))
