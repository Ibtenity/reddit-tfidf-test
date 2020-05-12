'''
utils.py
--------
This file contains utility functions used in the proof of concept program for the project
'Computing TF-IDF Vectors for Subreddits', by
Ken Tjhia <hexken@my.yorku.ca>
Qijin Xu <jackxu@my.yorku.ca>
Ibrahim Suedan <isuedan@hotmail.com>
'''


def coordinateMatrixMultiply(leftMatrix, rightMatrix):
    """
    :param leftMatrix: CoordinateMatrix
    :param rightMatrix: CoordinateMatrix
    :return: PipelineRDD of the (row, col, val) tuples
    from Stefan_Fairphone
    https://stackoverflow.com/questions/45881580/pyspark-rdd-sparse-matrix-multiplication-from-scala-to-python,
    which is a python implementation of the approach found at
    https://medium.com/balabit-unsupervised/scalable-sparse-matrix-multiplication-in-apache-spark-c79e9ffc0703
    """
    left = leftMatrix.entries.map(lambda e: (e.j, (e.i, e.value)))
    right = rightMatrix.entries.map(lambda e: (e.i, (e.j, e.value)))
    productEntries = left \
        .join(right) \
        .map(lambda e: ((e[1][0][0], e[1][1][0]), (e[1][0][1] * e[1][1][1]))) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda e: (*e[0], e[1]))
    return productEntries
