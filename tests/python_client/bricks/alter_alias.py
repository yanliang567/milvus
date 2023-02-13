import sys
import os
import logging
from pymilvus import utility, connections, \
    Collection
import create_n_insert
import pymilvus.exceptions

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


if __name__ == '__main__':
    host = sys.argv[1]
    c_name = sys.argv[2]        # collection name is <c_name>_aa or <c_name>_bb
    alias_name = f"{c_name}_alias"   # alias mame
    port = 19530
    conn = connections.connect('default', host=host, port=port)

    file_handler = logging.FileHandler(filename=f"/tmp/search_alias_{alias_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)
    logger = logging.getLogger('LOGGER_NAME')

    # check and get the collection info
    has_aa = utility.has_collection(collection_name=f'{c_name}_aa')
    has_bb = utility.has_collection(collection_name=f'{c_name}_bb')

    collection_name = f'{c_name}_bb' if has_aa else f'{c_name}_aa'
    create_n_insert.create_n_insert(collection_name=collection_name,index_type='hnsw')
    terminate = f'{c_name}_aa' if has_aa else f'{c_name}_bb'

    # alter alias
    try:
        aliases = utility.list_aliases(collection_name=alias_name)
        if alias_name in aliases:
            utility.alter_alias(collection_name=collection_name, alias=alias_name)
        else:
            utility.create_alias(collection_name=collection_name, alias=alias_name)
    except pymilvus.exceptions.DescribeCollectionException as e:
        utility.create_alias(collection_name=collection_name, alias=alias_name)
    collection = Collection(alias_name)
    logging.info(f"collection alias altered: {collection.description}")
    if utility.has_collection(collection_name=terminate):
        Collection(name=terminate).drop()
        logging.info(f"collection terminate dropped")

    logging.info(f"alter alias completed")
