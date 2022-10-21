from pymilvus import connections
from utils import *


def task_1(data_size, host):
    """
    task_1:
        before reinstall: create collection create index and insert data, load and search
        after reinstall: get collection, load, search, create index, load, and search
    """
    prefix = "task_1_"
    connections.connect(host=host, port=19530, timeout=60)
    get_collections(prefix)
    create_index(prefix)
    load_and_search(prefix)
    create_collections_and_insert_data(prefix,data_size)
    create_index(prefix)
    load_and_search(prefix)


def task_2(data_size, host):
    """
    task_2:
        before reinstall: create collection, create index, insert data and create index,load and search
        after reinstall: get collection, load, search, insert data, create index, load, and search
    """
    prefix = "task_2_"
    connections.connect(host=host, port=19530, timeout=60)
    get_collections(prefix)
    create_index(prefix)
    load_and_search(prefix)
    create_collections_and_insert_data(prefix, data_size)
    create_index(prefix)
    load_and_search(prefix)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='config for deploy test')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='milvus server ip')
    parser.add_argument('--data_size', type=int, default=3000, help='data size')
    args = parser.parse_args()
    data_size = args.data_size
    host = args.host
    print(f"data_size: {data_size}")
    task_1(data_size, host)
    task_2(data_size, host)
