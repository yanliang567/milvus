import time
import sys
import random
import string

import logging
from pymilvus import connections, DataType, \
    Collection, FieldSchema, CollectionSchema

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

nb = 50000
dim = 128
auto_id = False
index_params_dict = {
    "HNSW": {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 8, "efConstruction": 96}},
    "DISKANN": {"index_type": "DISKANN", "metric_type": "IP", "params": {}}
}


def gen_unique_str(length=32):
    str_value = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    return str_value


def gen_remarks_data(length=32):
    data = "{'content_ocr': '1.同步课时练习 enter (4)如果 ding 73是 bigcirc 的15倍,下面算式正确的是。 enter A. ding 73 times 15= bigcirc enter B. ding 73 times bigcirc =15 enter C. bigcirc times 15= ding 73 enter D. bigcirc div 15= ding 73 enter (5)列竖式计算'43 times 24'时,'2 times 43得86,86表示。 enter A.86个十 enter B.86个百 enter C.86个一 enter D.86个千 enter 4.学校食堂买来12箱鸡蛋,每箱4层,每层18个。一共买来鸡蛋多少个? enter 5.检验检疫中心正在对一批标本进行检验,3个工作人员8小时一共检验了360个标本。平均 enter 每个工作人员每小时检验多少个标本? enter 6.学校给一年级新生订了一批运动服(如图),购买的各种规格及数量如下表。 enter 规格 enter 大号 enter 中号 enter 小号 enter 数量/套 enter 64 enter 92 enter 58 enter (1)购买大号运动服一共要多少钱? enter 大号:95元/套 enter 中号:85元/套 enter 小号:75元/套 enter (2)购买小号运动服比中号运动服少用多少钱? enter 7.空调厂第一车间生产一批空调,如果每小时生产60台,12小时可以完成。现在为了赶工,要 enter 在9小时内完成,平均每小时要多生产多少台? enter 34'}"
    return gen_unique_str(length=length) + data


if __name__ == '__main__':
    host = sys.argv[1]  # host address
    collection_name = str(sys.argv[2])  # collection name
    index_type = str(sys.argv[3])  # index type
    shards = 2          # shards number
    insert_times = 60   # insert times

    port = 19530
    log_name = f"prepare_{collection_name}"

    file_handler = logging.FileHandler(filename=f"/tmp/{log_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    logging.info("start")
    connections.add_connection(default={"host": host, "port": 19530})
    connections.connect('default')

    field_id = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=72, description="string primary id")
    field_remarks = FieldSchema(name="remarks", dtype=DataType.VARCHAR, max_length=65535, description="string remarks")
    field_embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[field_id, field_remarks, field_embedding],
                              auto_id=auto_id, primary_field=field_id.name,
                              description=f"{collection_name}")
    collection = Collection(name=collection_name, schema=schema, shards_num=shards)
    logging.info(f"create {collection_name} successfully")

    index_params = index_params_dict.get(index_type.upper(), None)
    if index_params is None:
        logging.error(f"index type {index_type} no supported")
        exit(1)

    for i in range(insert_times):
        # prepare data
        ids = [gen_unique_str(length=32) for _ in range(nb)]
        remarks = [gen_remarks_data(length=128) for _ in range(nb)]
        embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
        data = [ids, remarks, embeddings]
        t0 = time.time()
        collection.insert(data)
        tt = round(time.time() - t0, 3)
        logging.info(f"insert {i} costs {tt}")

    collection.flush()
    logging.info(f"collection entities: {collection.num_entities}")

    if not collection.has_index():
        t0 = time.time()
        collection.create_index(field_name=field_embedding.name, index_params=index_params)
        tt = round(time.time() - t0, 3)
        logging.info(f"build index {index_params} costs {tt}")
    else:
        idx = collection.index()
        logging.info(f"index {idx.params} already exists")

    collection.load()
    logging.info("collection prepared completed")



