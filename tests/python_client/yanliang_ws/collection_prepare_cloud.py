import time
import sys
import random

import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

nb = 1000
dim = 128
auto_id = True
index_params = {"index_type": "AUTOINDEX", "params": {}, "metric_type": "L2"}

if __name__ == '__main__':
    uri = str(sys.argv[1])  # uri for milvus cloud instance
    user = str(sys.argv[2])  # user
    pwd = str(sys.argv[3])   # password
    collection_name = "autoindex_l2"   # str(sys.argv[4])  # collection name
    shards = 2          # shards number
    insert_times = 2000   # insert times

    # connections.add_connection()
    connections.connect(alias="default", uri=uri,
                        user=user, password=pwd, secure=True)
    log_name = f"prepare_{collection_name}"

    logging.basicConfig(filename=f"/tmp/{log_name}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.info("start")

    id = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
    age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id, age_field, embedding_field],
                              auto_id=auto_id, primary_field=id.name,
                              description=f"{collection_name}")
    collection = Collection(name=collection_name, schema=schema, shards_num=shards)
    logging.info(f"create {collection_name} successfully")

    for i in range(insert_times):
        # prepare data
        ages = [random.randint(1, 100) for _ in range(nb)]
        embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
        data = [ages, embeddings]
        t0 = time.time()
        collection.insert(data)
        tt = round(time.time() - t0, 3)
        logging.info(f"Insert {i} costs {tt}")

    collection.flush()
    logging.info(f"collection entities: {collection.num_entities}")

    t0 = time.time()
    collection.create_index(field_name=embedding_field.name, index_params=index_params)
    tt = round(time.time() - t0, 3)
    logging.info(f"Build index costs {tt}")

    collection.load()
    logging.info("Collection prepared completed")



