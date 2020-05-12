"""
transformers.py
----------------
This file contains the class definitions of several transformers used in the ETL pipeline of the
proof of concept program for the project 'Computing TF-IDF Vectors for Subreddits', by
Ken Tjhia <hexken@my.yorku.ca>
Qijin Xu <jackxu@my.yorku.ca>
Ibrahim Suedan <isuedan@hotmail.com>
"""
from pyspark.sql.functions import concat_ws, collect_list, udf, length, col, regexp_replace, lower
from pyspark.sql.types import *
from pyspark.ml import Transformer
from pyspark.ml.feature import Normalizer
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, CoordinateMatrix
from numpy import delete
from utils import coordinateMatrixMultiply


class Extractor(Transformer):
    """
    Concatenate all of the comment strings belonging to each subreddit into a big string
    """

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val

    def transform(self, df):
        return df.groupby(self.key).agg(concat_ws(" ", collect_list(self.val)).alias(self.outputCol))

    def getOutputCol(self):
        return self.outputCol

    def getinputCol(self):
        return self.inputCol


class Filterer(Transformer):
    """
    Filter out the subreddits whose 'document' string is less than args.minlength
    """

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None, minlength=None):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val
        self.minlength = minlength

    def transform(self, df):
        return df.filter((length(self.outputCol)) >= self.minlength)

    def getOutputCol(self):
        return self.outputCol

    def getinputCol(self):
        return self.inputCol


class Cleaner(Transformer):
    """
    Remove all non whitespace or non alphabetical characters
    """

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val

    def transform(self, df):
        return df.select(self.key, (lower(regexp_replace(self.val, "[^a-zA-Z\\s]", "")).alias(self.outputCol)))

    def getOutputCol(self):
        return self.outputCol

    def getinputCol(self):
        return self.inputCol


class TopKWords(Transformer):
    """
    find the k words with greatest tf-idf for each subreddit
    """

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None, vocab=None, nwords=5):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val
        self.vocab = vocab
        self.nwords = nwords

    def getOutputCol(self):
        return self.outputCol

    def getinputCol(self):
        return self.inputCol

    def transform(self, df):
        words_schema = StructType([
            StructField('tfidfs', ArrayType(FloatType()), nullable=False),
            StructField('words', ArrayType(StringType()), nullable=False)
        ])

        def getTopKWords(x, k=5):
            tfidfs = x.toArray()
            indices = tfidfs.argsort()[-k:][::-1]
            return tfidfs[indices].tolist(), [self.vocab[i] for i in indices]

        topkwords_udf = udf(lambda x: getTopKWords(x, k=self.nwords), words_schema)

        return df.withColumn('top_words', topkwords_udf(col('tfidf')))


class CosineSimilarity(Transformer):
    """
    Compute the cosine similarity between tfidf vectors of all subreddit pairs
    """

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None, spark=None):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val
        self.spark = spark

    def getOutputCol(self):
        return self.outputCol

    def getinputCol(self):
        return self.inputCol

    def transform(self, df):
        # add a row index, starting from 0 (to be used for matrix computations, i.e. cosine similarity)
        df.createOrReplaceTempView('data')
        data = self.spark.sql('select row_number() over (order by "subreddit") as index, * from data')
        df = data.withColumn('index', col('index') - 1)
        # normalize each tfidf vector to be unit length
        normalizer = Normalizer(inputCol="tfidf", outputCol="norm")
        df = normalizer.transform(df)
        # compute matrix of tfidf cosine similarities, all distributed :D (why we use BlockMatrix)
        # mat = IndexedRowMatrix(df.select('index', 'norm') \
        #                        .rdd.map(lambda x: IndexedRow(x['index'], x['norm'].toArray()))).toBlockMatrix()
        # dot = mat.multiply(mat.transpose())
        # cossimDF = dot.toIndexedRowMatrix().rows.toDF().withColumnRenamed('vector', 'cos_sims')
        mat = IndexedRowMatrix(df.select('index', 'norm') \
                               .rdd.map(lambda x: IndexedRow(x['index'], x['norm'].toArray()))).toCoordinateMatrix()
        cossim = CoordinateMatrix(coordinateMatrixMultiply(mat, mat.transpose())).toIndexedRowMatrix()
        cossimDF = cossim.rows.toDF().withColumnRenamed('vector', 'cos_sims')

        return df.join(cossimDF, ['index'])


class TopKSubreddits(Transformer):
    """
    For each subreddit, find the k other subreddits with greatest cosine similarity (of tf-idf vectors)
    """

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None, nsubreddits=5):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val
        self.nsubreddits = nsubreddits

    def getOutputCol(self):
        return self.outputCol

    def getinputCol(self):
        return self.inputCol

    def transform(self, df):
        subreddits_schema = StructType([
            StructField('cos_sims', ArrayType(FloatType()), nullable=False),
            StructField('subreddits', ArrayType(StringType()), nullable=False)
        ])

        # index_map is going to be in the driver local memory, it's generally not too big
        index_map = df.select('index', 'subreddit').toPandas().set_index('index')['subreddit'].to_dict()

        def getTopKSubreddits(x, k=5):
            # so we can skip the obvious most similar subreddit (itself)
            k += 1
            cos_sims = x.toArray()
            indices = cos_sims.argsort()[-k:][::-1]
            indices = delete(indices, 0)  # delete that first element which is the subreddit itself
            return cos_sims[indices].tolist(), [index_map[i] for i in indices]

        topksubreddits_udf = udf(lambda x: getTopKSubreddits(x, k=self.nsubreddits), subreddits_schema)

        return df.withColumn('top_subreddits', topksubreddits_udf(col('cos_sims')))
