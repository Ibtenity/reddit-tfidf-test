---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python [conda env:ml] *
    language: python
    name: conda-env-ml-py
---

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, collect_list, udf, length, col, regexp_replace, lower
from pyspark.sql.types import *

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Normalizer

from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRow, IndexedRowMatrix, CoordinateMatrix

from numpy import delete

spark = SparkSession\
    .builder\
    .enableHiveSupport()\
    .appName("tfidf-reddit-pipeline")\
    .getOrCreate()
```

```python
path100K = "RC100K_2019-07"
path300K = "RC300K_2019-07"
path1M = "RC1M_2019-07"
path5M = "RC5M_2019-07"
path10M = "RC10M_2019-07"
path15M = "RC15M_2019-07"

nwords=100
nsubreddits=100
mindf=1.0
minlength=50000
vocabsize=20000


path = path10M
df = spark.read.json(path + '.txt')
#commentsDF = spark.read.json(path100K)
```

```python
# from Stefan_Fairphone 
# https://stackoverflow.com/questions/45881580/pyspark-rdd-sparse-matrix-multiplication-from-scala-to-python
def coordinateMatrixMultiply(leftmatrix, rightmatrix):
    left  =  leftmatrix.entries.map(lambda e: (e.j, (e.i, e.value)))
    right = rightmatrix.entries.map(lambda e: (e.i, (e.j, e.value)))
    productEntries = left \
        .join(right) \
        .map(lambda e: ((e[1][0][0], e[1][1][0]), (e[1][0][1]*e[1][1][1]))) \
        .reduceByKey(lambda x,y: x+y) \
        .map(lambda e: (*e[0], e[1]))
    return productEntries
```

```python
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

    def __init__(self, key=None, val=None, inputCol=None, outputCol=None, nwords=5):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.key = key
        self.val = val
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
            return tfidfs[indices].tolist(), [vocab[i] for i in indices]

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

```

```python
# set up the ETL pipeline
extractor = Extractor(key='subreddit', val='body', inputCol='subreddit', outputCol='body')
cleaner = Cleaner(key='subreddit', val='body', inputCol=extractor.getOutputCol(), outputCol='body')
filterer = Filterer(key='subreddit', val='body', inputCol='subreddit', outputCol='body', minlength=minlength)
tokenizer = RegexTokenizer(inputCol=cleaner.getOutputCol(), outputCol="tokens", pattern="\\W")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="swr_tokens")
cv = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="tf", minDF=mindf, vocabSize=vocabsize)
idf = IDF(inputCol=cv.getOutputCol(), outputCol="tfidf")
topkwords = TopKWords(inputCol=idf.getOutputCol(), outputCol='top_words', nwords=nwords)
cos_similarity = CosineSimilarity(inputCol='subreddit', outputCol='norm', spark=spark)
topksubreddits = TopKSubreddits(inputCol=cos_similarity.getOutputCol(), outputCol='top_subreddits',
                                nsubreddits=nsubreddits)

pipeline = Pipeline(stages=[extractor, cleaner, filterer, tokenizer, remover, cv, idf, topkwords \
    , cos_similarity, topksubreddits])
```

```python
# fit the model, thene extract the learned vocabulary
model = pipeline.fit(df)
stages = model.stages
vectorizers = [s for s in stages if isinstance(s, CountVectorizerModel)]
vocab = vectorizers[0].vocabulary
# compute the tfidfs
df = model.transform(df)
df = df.drop('body', 'tf', 'tokens', 'swr_tokens', 'norm')
```

```python
trans_table = str.maketrans('', '', '-_')
df.write.mode('overwrite').saveAsTable(path.translate(trans_table))
```
