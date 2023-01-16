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


if __name__ == '__main__':
    host = sys.argv[1]
    collection_name = sys.argv[2]              # collection mame
    nb = int(sys.argv[3])                      # batch size of one insert request
    insert_interval = int(sys.argv[4])         # frequency of insert in seconds
    timeout = int(sys.argv[5])                 # insert timeout

    port = 19530

    file_handler = logging.FileHandler(filename=f"/tmp/insert_slowly_{collection_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    conn = connections.connect('default', host=host, port=port)
    if insert_interval <= 0:
        insert_interval = 60
        logging.info(f"set insert_interval to default: 60")
    if nb <= 0:
        insert_interval = 1
        logging.info(f"set insert_interval to default: 1")

    # check and get the collection info
    if not utility.has_collection(collection_name=collection_name):
        dim = 128
        auto_id = True
        shards = 2
        id_field = FieldSchema(name="id", dtype=DataType.INT64, description="auto primary id")
        age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        schema = CollectionSchema(fields=[id_field, age_field, embedding_field],
                                  auto_id=auto_id, primary_field=id_field.name,
                                  description=f"{collection_name}")
        collection = Collection(name=collection_name, schema=schema, shards_num=shards)
        logging.info(f"create {collection_name} successfully")

    c = Collection(name=collection_name)
    num_entities = c.num_entities
    logging.info(f"{collection_name} num_entities {num_entities}")

    logging.info(f"start to insert with nb={nb}, interval={insert_interval}, timeout={timeout}")
    start_time = time.time()
    r = 0
    while time.time() < start_time + timeout:
        # insert  entities
        data = gen_data_by_collection(collection=c, nb=nb, r=r)
        t1 = time.time()
        c.insert(data)
        t2 = round(time.time() - t1, 3)
        logging.info(f"{c.name} insert slowly in {t2}")
        r += 1
        time.sleep(insert_interval)

    logging.info(f"{collection_name} insert slowly completed")


