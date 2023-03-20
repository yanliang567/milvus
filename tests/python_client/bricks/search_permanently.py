import time
import sys
import random
import numpy as np
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def search(collection, field_name, search_params, nq, topk, threads_num, timeout):
    threads_num = int(threads_num)
    interval_count = 1000

    def search_th(col, thread_no):
        search_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            t1 = time.time()
            try:
                col.search(data=search_vectors, anns_field=field_name,
                           param=search_params, limit=topk
                           )
            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {col.description} search {interval_count} times in thread{thread_no}: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []

    threads = []
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=search_th, args=(collection, i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        search_latency = []
        count = 0
        start_time = time.time()
        while time.time() < start_time + timeout:
            count += 1
            search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
            t1 = time.time()
            try:
                # logging.info(f"ignore_growing: {ignore_growing}")
                collection.search(data=search_vectors, anns_field=field_name,
                                  # output_fields="",
                                  param=search_params, limit=topk
                                  )

            except Exception as e:
                logging.error(e)
            t2 = round(time.time() - t1, 4)
            search_latency.append(t2)
            if count == interval_count:
                total = round(np.sum(search_latency), 4)
                p99 = round(np.percentile(search_latency, 99), 4)
                avg = round(np.mean(search_latency), 4)
                qps = round(interval_count / total, 4)
                logging.info(f"collection {collection.description} search {interval_count} times single thread: cost {total}, qps {qps}, avg {avg}, p99 {p99} ")
                count = 0
                search_latency = []


if __name__ == '__main__':
    host = sys.argv[1]
    name = sys.argv[2]                  # collection mame/alias
    th = int(sys.argv[3])               # search thread num
    timeout = int(sys.argv[4])          # search timeout, permanently if 0
    ignore_growing = bool(sys.argv[5])  # ignore_growing, default False
    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/search_{name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)

    nq = 1
    topk = 1
    ef = 32
    nprobe = 16

    # check and get the collection info
    if not utility.has_collection(collection_name=name):
        logging.error(f"collection: {name} does not exit")
        exit(0)

    collection = Collection(name=name)
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            vector_field_name = field.name
            dim = field.params.get("dim")
            break

    if not collection.has_index():
        logging.error(f"collection: {name} has no index")
        exit(0)
    idx = collection.index()
    metric_type = idx.params.get("metric_type")
    index_type = idx.params.get("index_type")
    if index_type == "HNSW":
        search_params = {"metric_type": metric_type, "params": {"ef": ef}, "ignore_growing": ignore_growing}
    elif index_type in ["IVF_SQ8", "IVF_FLAT"]:
        search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}, "ignore_growing": ignore_growing}
    elif index_type == "DISKANN":
        search_params = {"metric_type": metric_type, "params": {"search_list": 100}, "ignore_growing": ignore_growing}
    else:
        logging.error(f"index: {index_type} does not support yet")
        exit(0)

    logging.info(f"index param: {idx.params}")
    logging.info(f"search_param: {search_params}")

    # flush before indexing
    t1 = time.time()
    num = collection.num_entities
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert {name} flushed num_entities {num}: {t2}")

    logging.info(utility.index_building_progress(name))

    # load collection
    t1 = time.time()
    collection.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert load {name}: {t2}")

    logging.info(f"search start: nq{nq}_top{topk}_threads{th}")
    search(collection, vector_field_name, search_params, nq, topk, th, timeout)
    logging.info(f"search completed ")

