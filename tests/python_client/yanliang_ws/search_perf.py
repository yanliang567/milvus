import time
import sys
import random

import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

field_name = "embedding"
search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
index_params = {"index_type": "IVF_SQ8", "params": {"nlist": 1024}, "metric_type": "L2"}
nqs = [1, 10, 100, 1000]
topks = [1, 10, 100, 1000]
dim = 128


def search(name, nq, topk, threads_num, times_per_thread):

    # init
    nq = int(nq)
    topk = int(topk)
    threads_num = int(threads_num)
    times_per_thread = int(times_per_thread)
    collection = Collection(name=name)

    # build index
    t1 = time.time()
    collection.create_index(field_name=field_name, index_params=index_params)
    t2 = time.time() - t1
    logging.info(f"assert build index {name}: {t2}")

    # load collection
    collection.load()
    t1 = time.time()
    collection.load()
    t2 = time.time() - t1
    logging.info(f"assert load {name}: {t2}")

    search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]

    def search_th(col, rounds, thread_no):
        for r in range(rounds):
            t1 = time.time()
            col.search(data=search_vectors, anns_field=field_name,
                       search_params=search_params, limit=topk)
            t2 = time.time() - t1
            logging.info(f"assert search thread{thread_no} round{r}: {t2}")

    threads = []
    logging.info(f"ready to search {name}, nq={nq}, topk={topk} for {times_per_thread} times per thread")
    logging.info(f"index param: {index_params}, search_param: {search_params}")
    search_t1 = time.time()
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=search_th, args=(collection, int(times_per_thread), i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        for r in range(times_per_thread):
            t1 = time.time()
            collection.search(data=search_vectors, anns_field=field_name,
                              search_params=search_params, limit=topk)
            t2 = time.time() - t1
            logging.info(f"assert search thread0 round{r}: {t2}")
    search_t2 = time.time() - search_t1
    logging.info(f"search cost {search_t2}, nq={nq}, topk={topk}, threads={times_per_thread}")

    # collection.drop()
    # log.info("collection dropped.")


if __name__ == '__main__':

    th = int(sys.argv[1])       # insert thread num
    per_thread = int(sys.argv[2])     # times per thread
    host = "10.98.0.26"
    port = 19530
    conn = connections.connect('default', host=host, port=port)
    logging.basicConfig(filename=f"/tmp/search_threads{th}_per{per_thread}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    collection_name = f"insert_nb80000_shards4_threads10_per10"   #8M

    for nq in nqs:
        for topk in topks:
            t1 = time.time()
            search(collection_name, nq, topk, th, per_thread)
            t2 = time.time() - t1
            logging.info(f"search cost total {t2}, nq={nq}, topk={topk}, threads={per_thread}")


