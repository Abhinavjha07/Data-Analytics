import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"
import findspark
import re
from pyspark.sql import SparkSession

findspark.init()
spark = SparkSession.builder.master("local[*]").getOrCreate()



def computeContribs(urls, rank):
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parseNeighbors(urls):
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]


lines = spark.read.text('/content/graphs/small.txt').rdd.map(lambda r: r[0])
links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

N = ranks.count()
iterations = 40

for iteration in range(iterations):
    contribs = links.join(ranks).flatMap(lambda urls_rank: computeContribs(urls_rank[1][0], urls_rank[1][1]))
    
    ranks = contribs.reduceByKey(lambda r1,r2 : r1 + r2).mapValues(lambda rank: rank * 0.80 + 0.20*(1/(N)))


rank_list = {}
for (link, rank) in ranks.collect():
    rank_list[link] = rank
    print("%s has rank: %s." % (link, rank))

spark.stop()


ranks = dict(sorted(rank_list.items(), key=lambda kv: kv[1],reverse = True))

x = 0
print('Top 5 : \nNode\t Rank')
for k in ranks:
    x += 1
    print(k, ranks[k])
    if(x == 5):
        break

ranks = dict(sorted(rank_list.items(), key=lambda kv: kv[1]))

x = 0
print('Bottom 5 : \nNode\t Rank')
for k in ranks:
    x += 1
    print(k, ranks[k])
    if(x == 5):
        break
