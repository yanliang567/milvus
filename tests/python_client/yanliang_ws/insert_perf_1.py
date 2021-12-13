import time
import sys
import random
from sklearn import preprocessing
import threading
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

prefix = "ins_"
# nbs = [1, 100, 1000, 10*1000, 20*1000, 40*1000, 60*1000, 80*1000, 100*1000]
nb = 50000
dim = 128
auto_id = False
build = True
# index_params = {"index_type": "IVF_SQ8", "params": {"nlist": 1024}, "metric_type": "L2"}
# index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 2048}, "metric_type": "L2"}
index_params = {"index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}, "metric_type": "L2"}


def do_insert(conn, data, threads_num, ins_times_per_thread, name):

    def insert_th(conn, data, rounds, thread_no):
        for r in range(rounds):
            t1 = time.time()
            res = conn.insert(name, data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert thread{thread_no} round{r}: {t2}")

    # insert
    threads = []
    logging.info(f"ready to insert {name}, insert {ins_times_per_thread} times per thread")
    if threads_num > 1:
        for i in range(threads_num):
            t = threading.Thread(target=insert_th, args=(conn, data, int(ins_times_per_thread), i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        for r in range(ins_times_per_thread):
            t1 = time.time()
            res = conn.insert(name, data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert thread0 round{r}: {t2}")


def gen_vectors(num, dim, is_normal=True):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


if __name__ == '__main__':
    host = sys.argv[1]  # host address
    shards = int(sys.argv[2])  # shards number
    th = int(sys.argv[3])  # insert thread num
    per_thread = int(sys.argv[4])  # insert times per thread

    port = 19530
    conn = connections.connect('default', host=host, port=port)
    log_name = f"insert_shards{shards}_threads{th}_per{per_thread}"

    logging.basicConfig(filename=f"/tmp/{log_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info("Insert perf ... ...")

    fields = [
        {"name": "id", "type": DataType.INT64, "is_primary": True},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}}
    ]
    create_param = {"fields": fields, "auto_id": auto_id}
    params = {}
    params.update({"shards_num": shards})

    collection_name = f"insert_nb{nb}_shards{shards}_threads{th}_per{per_thread}_f"
    conn.create_collection(collection_name, create_param, **params)

    # prepare data
    vectors = gen_vectors(nb, dim, False)
    entities = [
        {"name": "id", "type": DataType.INT64, "values": [i for i in range(nb)]},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "values": vectors}
    ]

    if build is True:
        conn.create_index(collection_name=collection_name,
                          field_name="embedding", params=index_params)

    t1 = time.time()
    do_insert(conn, entities, th, per_thread, collection_name)
    t2 = time.time() - t1
    req_per_sec = round(per_thread * th / t2, 3)  # how many insert requests response per second
    entities_throughput = round(nb * req_per_sec, 3)  # how many entities inserted per second
    logging.info(f"Insert  {collection_name} cost {round(t2, 3)}, "
                 f"req_per_second {req_per_sec}, entities_throughput {entities_throughput}")
