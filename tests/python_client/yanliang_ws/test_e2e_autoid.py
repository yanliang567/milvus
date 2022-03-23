import sys
import time
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

prefix = "e2e_"
cus_index = {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 500}, "metric_type": "IP"}
search_params = {"params": {"ef": 64}, "metric_type": "IP"}
# cus_index = {"index_type": "IVF_SQ8", "params": {"nlist": 1024}, "metric_type": "L2"}
# search_params = {"metric_type": "L2", "params": {"nprobe": 8}}
dim = 1024
nb = 50000
insert_rounds = 20
nq = 1
topK = 1
auto_id = False
build = True
expr = False
search_rounds = 100


def create_collection(name):
    t0 = time.time()
    collection_name = name
    if auto_id is True:
        id = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
        age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[id, age_field, embedding_field],
                                  auto_id=True, primary_field=id.name,
                                  description="collection of xxx")
        collection = Collection(name=collection_name, schema=schema)
        tt = time.time() - t0
        logging.info(f"assert create {collection_name}: {tt}")
    else:
        id = FieldSchema(name="id", dtype=DataType.INT64, description="custom primary id")
        age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[id, age_field, embedding_field],
                                  auto_id=False, primary_field=id.name,
                                  description="collection of custom id")
        collection = Collection(name=collection_name, schema=schema)
        tt = time.time() - t0
        logging.info(f"assert create {collection_name}: {tt}")

    if build is True:
        collection.create_index(field_name=embedding_field.name, index_params=cus_index)

    # insert
    if auto_id is True:
        pass
    else:
        pks = []
        for r in range(insert_rounds):
            ids = [i for i in range(r*nb, (r+1)*nb)]
            ages = [random.randint(1, 100) for i in range(nb)]
            embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
            data = [ids, ages, embeddings]
            t1 = time.time()
            res = collection.insert(data)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert insert round{r}: {t2}")
            pks.extend(res.primary_keys)
        logging.info(f"total pks: {len(pks)}")

    return collection


def do_search(collection):
    # flush
    t1 = time.time()
    num = collection.num_entities
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert flush {num} entities in {t2} seconds ")

    # build index again
    logging.info(f"index params: {cus_index}")
    t1 = time.time()
    collection.create_index(field_name="embedding", index_params=cus_index)
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert build index in {t2} seconds ")

    # load
    t1 = time.time()
    collection.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"assert load time: {t2}")

    # search
    logging.info(f"search params: {search_params}, nq={nq}, topk={topK}")
    pk_ids = [i for i in range(0, 5000)]
    for i in range(search_rounds):
        search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
        if expr is True:
            t1 = time.time()
            collection.search(data=search_vectors, anns_field="embedding",
                              param=search_params, limit=topK,
                              expr=f'id in {pk_ids}')
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert search with expr round{i} in time: {t2}")
        else:
            t1 = time.time()
            collection.search(data=search_vectors, anns_field="embedding",
                              param=search_params, limit=topK)
            t2 = round(time.time() - t1, 3)
            logging.info(f"assert search round{i} in time: {t2}")


if __name__ == '__main__':
    host = sys.argv[1]  # host address
    collection_name = sys.argv[2]  # collection_name

    port = 19530
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')
    log_name = f"{collection_name}_search"
    logging.basicConfig(filename=f"/tmp/{log_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    if utility.has_collection(collection_name=collection_name):
        collection = Collection(collection_name)
    else:
        collection = create_collection(collection_name)
    do_search(collection=collection)
