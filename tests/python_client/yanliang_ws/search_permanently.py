import time
import sys
import random

import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def search(collection, field_name, search_params, nq, topk, threads_num, timeout):
    threads_num = int(threads_num)

    def search_th(col, thread_no):
        start_time = time.time()
        while time.time() < start_time + timeout:
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            t1 = time.time()
            col.search(data=search_vectors, anns_field=field_name,
                       param=search_params, limit=topk)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert search thread{thread_no}: {t2}")

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=search_th, args=(collection, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        start_time = time.time()
        while time.time() < start_time + timeout:
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            t1 = time.time()
            collection.search(data=search_vectors, anns_field=field_name,
                              param=search_params, limit=topk)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert single thread search: {t2}")


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]       # collection mame
    th = int(sys.argv[3])               # search thread num
    timeout = int(sys.argv[4])          # search timeout, permanently if 0
    port = 19530
    conn = connections.connect('default', host=host, port=port)

    logging.basicConfig(filename=f"/tmp/{collection_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    nq = 1
    topk = 1
    ef = 32
    nprobe = 16

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        logging.error(f"collection: {collection_name} does not exit")
        exit(0)

    collection = Collection(name=collection_name)
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == 101:
            vector_field_name = field.name
            dim = field.params.get("dim")
            break

    if not collection.has_index():
        logging.error(f"collection: {collection_name} has no index")
        exit(0)
    idx = collection.index()
    metric_type = idx.params.get("metric_type")
    index_type = idx.params.get("index_type")
    if index_type == "HNSW":
        search_params = {"metric_type": metric_type, "params": {"ef": ef}}
    elif index_type in ["IVF_SQ8", "IVF_FLAT"]:
        search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    else:
        logging.error(f"index: {index_type} does not support yet")
        exit(0)

    logging.info(f"index param: {idx.params}")
    logging.info(f"search_param: {search_params}")

    # flush before indexing
    t1 = time.time()
    num = collection.num_entities
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert {collection_name} flushed num_entities {num}: {t2}")

    logging.info(utility.index_building_progress(collection_name))

    # load collection
    t1 = time.time()
    collection.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert load {collection_name}: {t2}")

    logging.info(f"search start: nq{nq}_top{topk}_threads{th}")
    search(collection, vector_field_name, search_params, nq, topk, th, timeout)
    logging.info(f"search completed ")

