import time
import sys
import random

import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

prefix = "ins_"
nbs = [100, 1000, 10*1000, 20*1000, 40*1000, 60*1000, 100*1000]
# threads_num = 10
shardNum = 2
dim = 128


def insert(nb1, total_entities1, threads_num, name):

    # create
    nb = int(nb1)
    total_entities = int(total_entities1)
    threads_num = int(threads_num)
    t0 = time.time()
    auto_id = FieldSchema(name="auto_id", dtype=DataType.INT64, description="auto primary id")
    age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[auto_id, age_field, embedding_field],
                              auto_id=True, primary_field=auto_id.name,
                              description="collection of insert perf")
    collection = Collection(name=name, schema=schema, shards_num=shardNum)
    tt = time.time() - t0
    logging.info(f"assert create nb{nb} collection: {tt}")

    ages = [random.randint(1, 100) for i in range(nb)]
    embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
    data = [ages, embeddings]

    def insert_th(col_w, data, rounds, thread_no):
        for r in range(rounds):
            t1 = time.time()
            res = col_w.insert(data)
            t2 = time.time() - t1
            logging.info(f"assert insert thread{thread_no} round{r}: {t2}")

    # insert
    threads = []
    ins_times = total_entities / nb
    ins_per_thread = ins_times / threads_num
    logging.info(f"ready to insert {total_entities} entities by {ins_times} insert times "
                 f"divided in {threads_num} threads, and {ins_per_thread} rounds per thread")
    for i in range(threads_num):
        t = threading.Thread(target=insert_th, args=(collection, data, int(ins_per_thread), i))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # collection.drop()
    # log.info("collection dropped.")


if __name__ == '__main__':
    nb1 = sys.argv[1]
    entities1 = sys.argv[2]
    threads = sys.argv[3]
    logging.basicConfig(filename=f"/tmp/insert_nb{nb1}_total{entities1}_threads{threads}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info(f"input: nb={nb1}, entities={entities1}, threads={threads}")
    host = "10.98.0.20"
    port = 19530
    conn = connections.connect('default', host=host, port=port)
    collection_name = f"insert_nb{nb1}_total{entities1}_threads{threads}"
    t1 = time.time()
    insert(nb1, entities1, threads, collection_name)
    t2 = time.time() - t1
    logging.info(f"Insert cost {t2}, {collection_name}")
