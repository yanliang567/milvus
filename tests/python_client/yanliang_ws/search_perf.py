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
search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
index_params = {"index_type": "IVF_SQ8", "params": {"nlist": 1024}, "metric_type": "L2"}
# index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 2048}, "metric_type": "L2"}
# index_params = {"index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}, "metric_type": "L2"}
# search_params = {"metric_type": "L2", "params": {"ef": 64}}
nqs = [1]
topks = [10]
dim = 128


def search(collection, search_vectors, topk, threads_num, times_per_thread):

    # init
    # nq = len(search_vectors)
    topk = int(topk)
    threads_num = int(threads_num)
    times_per_thread = int(times_per_thread)

    def search_th(col, rounds, thread_no):
        for r in range(rounds):
            t1 = time.time()
            col.search(data=search_vectors, anns_field=field_name,
                       param=search_params, limit=topk)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert search thread{thread_no} round{r}: {t2}")

    threads = []
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
                              param=search_params, limit=topk)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert search thread0 round{r}: {t2}")

    # collection.drop()
    # log.info("collection dropped.")


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]       # collection mame
    th = int(sys.argv[3])               # search thread num
    per_thread = int(sys.argv[4])       # times per thread
    port = 19530
    conn = connections.connect('default', host=host, port=port)

    logging.basicConfig(filename=f"/tmp/{collection_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info("search perf test....")

    # build index
    collection = Collection(name=collection_name)
    logging.info(f"index param: {index_params}")
    logging.info(f"search_param: {search_params}")

    # flush before indexing
    t1 = time.time()
    num = collection.num_entities
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert {collection_name} flushed num_entities {num}: {t2}")

    t1 = time.time()
    collection.create_index(field_name=field_name, index_params=index_params)
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert build index {collection_name}: {t2}")
    logging.info(utility.index_building_progress(collection_name))

    # load collection
    t1 = time.time()
    collection.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert load {collection_name}: {t2}")

    for nq in nqs:
        for topk in topks:
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]

            logging.info(f"Start search nq{nq}_top{topk}_{th}threads_per{per_thread}")
            t1 = time.time()
            search(collection, search_vectors, topk, th, per_thread)
            t2 = time.time() - t1
            query_per_sec = round(th * per_thread / t2, 3)       # how many search requests response per second
            vectors_throughput = round(nq * query_per_sec, 3)    # how many vectors be searched per second
            logging.info(f"Compete search nq{nq}_top{topk}_{th}threads_per{per_thread}, "
                         f"cost {t2}, QPS: {query_per_sec}, vectors_throughput: {vectors_throughput}")

    # collection.release()
    # collection.index().drop()
