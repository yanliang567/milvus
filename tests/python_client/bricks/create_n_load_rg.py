import time
import sys
import random

import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema, utility
import create_n_insert


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

default_rg = "__default_resource_group"

#
# def create_n_insert(collection_name, index_type, metric_type="L2"):
#     id_field = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
#     age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
#     embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
#     schema = CollectionSchema(fields=[id_field, age_field, embedding_field],
#                               auto_id=auto_id, primary_field=id_field.name,
#                               description=f"{collection_name}")
#     collection = Collection(name=collection_name, schema=schema, shards_num=shards)
#     logging.info(f"create {collection_name} successfully")
#
#     for i in range(insert_times):
#         # prepare data
#         ages = [random.randint(1, 100) for _ in range(nb)]
#         embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
#         data = [ages, embeddings]
#         t0 = time.time()
#         collection.insert(data)
#         tt = round(time.time() - t0, 3)
#         logging.info(f"insert {i} costs {tt}")
#
#     collection.flush()
#     logging.info(f"collection entities: {collection.num_entities}")
#
#     index_params_dict = {
#         "HNSW": {"index_type": "HNSW", "metric_type": metric_type, "params": {"M": 8, "efConstruction": 96}},
#         "DISKANN": {"index_type": "DISKANN", "metric_type": metric_type, "params": {}}
#     }
#     index_params = index_params_dict.get(index_type.upper(), None)
#     if index_params is None:
#         logging.error(f"index type {index_type} no supported")
#         exit(1)
#
#     if not collection.has_index():
#         t0 = time.time()
#         collection.create_index(field_name=embedding_field.name, index_params=index_params)
#         tt = round(time.time() - t0, 3)
#         logging.info(f"build index {index_params} costs {tt}")
#     else:
#         idx = collection.index()
#         logging.info(f"index {idx.params} already exists")
#

if __name__ == '__main__':
    host = sys.argv[1]  # host address
    name = str(sys.argv[2])  # collection name
    index = str(sys.argv[3])  # index type
    metric = str(sys.argv[4])  # metric type
    rg_name = str(sys.argv[5])  # resource group name
    port = 19530
    log_name = f"prepare_{name}"

    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')

    default_rg_info = utility.describe_resource_group(name=default_rg)
    rgs = utility.list_resource_groups()
    if rg_name in rgs:
        logging.info(f"rg {rg_name} already exists")
    else:
        utility.create_resource_group(name=rg_name)
    rg_info = utility.describe_resource_group(name=rg_name)
    if rg_info.num_available_node < 1:
        if default_rg_info.num_available_node < 1:
            logging.error(f"there is no available nodes in default rg")
            exit(0)
        utility.transfer_node(source=default_rg, target=rg_name, num_node=1)

    # create an insert
    create_n_insert.create_n_insert(collection_name=name, index_type=index, metric_type=metric)

    # load collection
    c = Collection(name=name)
    c.load(_resource_groups=[rg_name])
    logging.info(f"collection loaded to resource groups: {[rg_name]}")

    logging.info("collection prepared completed")



