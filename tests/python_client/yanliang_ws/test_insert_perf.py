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
nbs = [1, 100, 1000, 10*1000, 20*1000, 40*1000, 60*1000, 80*1000, 100*1000]
dim = 128
auto_id = False


def insert(data, threads_num, ins_times_per_thread, collection):

    def insert_th(col_w, data, rounds, thread_no):
        for r in range(rounds):
            t1 = time.time()
            res = col_w.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert thread{thread_no} round{r}: {t2}")

    # insert
    threads = []
    logging.info(f"ready to insert {collection.name}, insert {ins_times_per_thread} times per thread")
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=insert_th, args=(collection, data, int(ins_times_per_thread), i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        for r in range(ins_times_per_thread):
            t1 = time.time()
            res = collection.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert thread0 round{r}: {t2}")

    return collection
    # collection.drop()
    # log.info("collection dropped.")


if __name__ == '__main__':
    shards = int(sys.argv[1])   # shards number
    th = int(sys.argv[2])       # insert thread num
    per_thread = int(sys.argv[3])     # insert times per thread

    host = "10.98.0.8"
    port = 19530
    conn = connections.connect('default', host=host, port=port)
    log_name = f"insert_shards{shards}_threads{th}_per{per_thread}"

    logging.basicConfig(filename=f"/tmp/{log_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info("Insert perf ... ...")

    for nb1 in nbs:
        collection_name = f"insert_nb{nb1}_shards{shards}_threads{th}_per{per_thread}"
        t0 = time.time()
        if auto_id is True:
            collection_name = f"insert_nb{nb1}_shards{shards}_threads{th}_per{per_thread}_t"
            id = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
            age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
            embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            schema = CollectionSchema(fields=[id, age_field, embedding_field],
                                      auto_id=True, primary_field=id.name,
                                      description="collection of insert perf")
            collection = Collection(name=collection_name, schema=schema, shards_num=shards)
            tt = time.time() - t0
            logging.info(f"assert create {collection_name}: {tt}")

            # prepare data
            ages = [random.randint(1, 100) for i in range(nb1)]
            embeddings = [[random.random() for _ in range(dim)] for _ in range(nb1)]
            data = [ages, embeddings]
        else:
            collection_name = f"insert_nb{nb1}_shards{shards}_threads{th}_per{per_thread}_f"
            id = FieldSchema(name="id", dtype=DataType.INT64, description="custom primary id")
            age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
            embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            schema = CollectionSchema(fields=[id, age_field, embedding_field],
                                      auto_id=False, primary_field=id.name,
                                      description="collection of insert perf")
            collection = Collection(name=collection_name, schema=schema, shards_num=shards)
            tt = time.time() - t0
            logging.info(f"assert create {collection_name}: {tt}")

            # prepare data
            ids = [i for i in range(nb1)]
            ages = [random.randint(1, 100) for i in range(nb1)]
            embeddings = [[random.random() for _ in range(dim)] for _ in range(nb1)]
            data = [ids, ages, embeddings]

        t1 = time.time()
        collection = insert(data, th, per_thread, collection)
        t2 = time.time() - t1
        req_per_sec = round(per_thread * th / t2, 3)            # how many insert requests response per second
        entities_throughput = round(nb1 * req_per_sec, 3)         # how many entities inserted per second
        logging.info(f"Insert  {collection_name} cost {round(t2, 3)}, "
                     f"req_per_second {req_per_sec}, entities_throughput {entities_throughput}")

        # t0 = time.time()
        # logging.info(f"collection {collection_name}, num_entities: {collection.num_entities}")
        # tt = time.time() - t0
        # logging.info(f"assert flush: {tt}")


