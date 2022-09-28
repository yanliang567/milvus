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
dim = 128
auto_id = True
shards = 2

if __name__ == '__main__':
    # host = sys.argv[1]
    # collection_alias = sys.argv[2]       # collection alias
    host = "10.100.31.105"
    collection_alias = "collection_alias"       # collection alias
    port = 19530
    conn = connections.connect('default', host=host, port=port)

    logging.basicConfig(filename=f"/tmp/{collection_alias}.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    collection_nameA = f"collection_a"
    id = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
    age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[id, age_field, embedding_field],
                              auto_id=auto_id, primary_field=id.name,
                              description="my collection aaa")
    collection = Collection(name=collection_nameA, schema=schema, shards_num=shards)
    logging.info(f"create {collection_nameA} successfully")
    collection_nameB = f"collection_b"
    schema = CollectionSchema(fields=[id, age_field, embedding_field],
                              auto_id=auto_id, primary_field=id.name,
                              description="my collection bbb")
    collection = Collection(name=collection_nameB, schema=schema, shards_num=shards)
    logging.info(f"create {collection_nameB} successfully")

    utility.create_alias(collection_name=collection_nameA, alias=collection_alias)
    c = Collection(collection_alias)
    c.description
