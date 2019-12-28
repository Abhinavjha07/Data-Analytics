import re
import sys
from pyspark import SparkConf, SparkContext
conf = SparkConf()

sc = SparkContext(conf = conf)
lines = sc.textFile(sys.argv[1])
bigrams = lines.map(lambda s : s.split(' ')).flatMap(lambda s : [((s[i],s[i+1]),1) for i in range(0, len(s)-1)])
bigrams.collect()

counts = bigrams.reduceByKey(lambda x, y : x+y)

counts.saveAsTextFile(sys.argv[2])
sc.stop()
