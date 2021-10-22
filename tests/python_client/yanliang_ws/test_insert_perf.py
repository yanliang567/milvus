import datetime
import time
import sys

import pytest
import threading
from time import sleep

from base.client_base import TestcaseBase
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel
from utils.util_log import test_log as log
from pymilvus import utility, connections, Collection

prefix = "ins_"
nbs = [100, 1000, 10*1000, 20*1000, 40*1000, 60*1000, 100*1000]
threads_num = 2
rounds = 2
shardNum = 2


def insert(entities):

    for nb in nbs:
        # create
        name = cf.gen_unique_str(prefix)
        t0 = time.time()
        schema = cf.gen_default_collection_schema(auto_id=True)
        collection = Collection(name=name, schema=schema, shards_num=shardNum)
        tt = time.time() - t0
        log.info(f"assert create nb{nb}: {tt}")

        def insert_th(col_w, entities, threads_n):
            for _ in range(threads_n):
                t1 = time.time()
                _, res = col_w.insert(entities)
                t2 = time.time() - t1
                log.info(f"assert insert round{r}: {t2}")

        # insert
        threads = []
        data = cf.gen_default_list_data(nb)[1:3]
        for r in range(rounds):
            for _ in range(threads_num):
                t = threading.Thread(target=insert_th, args=(collection, data, threads_num))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        collection.drop()
        log.info("collection dropped.")
        sleep(10)


if __name__ == '__main__':
    entities = sys.argv[1]

