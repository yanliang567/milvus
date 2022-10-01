import time
import sys
import random
import logging
from pymilvus import utility, connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


def get_dim(collection):
    fields = collection.schema.fields
    for field in fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
    return dim


def delete_entities(collection, nb, search_params, upsert_rounds):
    dim = get_dim(collection=collection)
    auto_id = collection.schema.auto_id
    primary_field_name = collection.primary_field.name
    vector_field_name = collection._get_vector_field()
    if auto_id:
        for r in range(upsert_rounds):
            search_vector = [[random.random() for _ in range(dim)] for _ in range(1)]
            results = c.search(data=search_vector, anns_field=vector_field_name,
                               param=search_params, limit=nb)
            for hits in results:
                ids = hits.ids
                c.delete(expr=f"{primary_field_name} in {ids}")
                logging.info(f"deleted {len(ids)} entities")
    else:
        for r in range(upsert_rounds):
            start_uid = r * nb
            end_uid = start_uid + nb
            c.delete(expr=f"{primary_field_name} in [{start_uid}, {end_uid}]")
            logging.info(f"deleted entities {start_uid}-{end_uid}")


def gen_data_by_collection(collection, nb, r):
    data = []
    start_uid = r * nb
    fields = collection.schema.fields
    auto_id = collection.schema.auto_id
    for field in fields:
        field_values = []
        if field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get("dim")
            field_values = [[random.random() for _ in range(dim)] for _ in range(nb)]
        if field.dtype == DataType.INT64:
            if field.is_primary:
                if not auto_id:
                    field_values = [_ for _ in range(start_uid, start_uid + nb)]
                else:
                    continue
            else:
                field_values = [_ for _ in range(nb)]
        if field.dtype == DataType.VARCHAR:
            field_values = [str(random.random()) for _ in range(nb)]
        if field.dtype == DataType.FLOAT:
            field_values = [random.random() for _ in range(nb)]
        data.append(field_values)
    return data


def insert_entities(collection, nb, upsert_rounds):
    for r in range(upsert_rounds):
        data = gen_data_by_collection(collection=collection, nb=nb, r=r)
        t1 = time.time()
        collection.insert(data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{collection.name} {r} upsert in {t2}")


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]              # collection mame
    # host = "10.100.31.105"
    # collection_name = "hnsw_ip_m8_ef96"            # collection mame
    delete_percent = 10          # percent of entities to be deleted
    insert_percent = 30
    port = 19530
    conn = connections.connect('default', host=host, port=port)

    logging.basicConfig(filename=f"/tmp/{collection_name}_upsert.log",
                        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    if delete_percent < 1 or delete_percent > 100:
        logging.error(f"delete percent shall be 1-100")
        exit(0)
    if insert_percent < 1 or insert_percent > 100:
        logging.error(f"insert percent shall be 1-100")
        exit(0)

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        logging.error(f"collection: {collection_name} does not exit")
        exit(0)

    c = Collection(name=collection_name)
    nb = 10000
    num_entities = c.num_entities
    if num_entities <= 0:
        logging.error(f"collection: {collection_name} num_entities is empty")
        exit(0)
    logging.info(f"{collection_name} num_entities {num_entities}")

    if not c.has_index():
        logging.error(f"collection: {collection_name} has no index")
        exit(0)
    idx = c.index()
    metric_type = idx.params.get("metric_type")
    index_type = idx.params.get("index_type")
    if index_type == "HNSW":
        search_params = {"metric_type": metric_type, "params": {"ef": nb}}
    elif index_type in ["IVF_SQ8", "IVF_FLAT"]:
        search_params = {"metric_type": metric_type, "params": {"nprobe": 32}}
    else:
        logging.error(f"index: {index_type} does not support yet")
        exit(0)

    # load collection
    t1 = time.time()
    c.load()
    t2 = round(time.time() - t1, 3)
    logging.info(f"load {collection_name}: {t2}")

    delete_num = num_entities * delete_percent // 100
    # insert nb if less than nb
    delete_rounds = int(delete_num // nb + 1)
    logging.info(f"{delete_rounds * nb} entities to be deleted in {delete_rounds} rounds")

    # delete xx% entities
    delete_entities(collection=c, nb=nb,
                    search_params=search_params,
                    upsert_rounds=delete_rounds)

    insert_num = num_entities * insert_percent // 100
    # insert nb if less than nb
    insert_rounds = int(insert_num // nb + 1)
    logging.info(f"{insert_rounds * nb} entities to be upsert in {insert_rounds} rounds")

    # insert xx% entities
    insert_entities(collection=c, nb=nb, upsert_rounds=insert_rounds)

    logging.info(f"{collection_name} upsert completed")


