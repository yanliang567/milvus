# import datetime
# import pytest
# import threading
# from time import sleep
#
# from base.client_base import TestcaseBase
# from common import common_func as cf
# from common import common_type as ct
# from common.common_type import CaseLabel
# from utils.util_log import test_log as log
# from pymilvus import utility
#
# prefix = "ins_"
# nbs = [100, 1000, 10*1000, 20*1000, 40*1000, 60*1000, 100*1000]
# threads_num = 2
# rounds = 2
#
#
# class TestInsertPerf(TestcaseBase):
#     """ Test case of end to end"""
#     @pytest.mark.tags(CaseLabel.L3)
#     def test_milvus_default(self):
#
#         for nb in nbs:
#             # create
#             name = cf.gen_unique_str(prefix)
#             t0 = datetime.datetime.now()
#             schema = cf.gen_default_collection_schema(auto_id=True)
#             collection_w = self.init_collection_wrap(name=name, schema=schema)
#             tt = datetime.datetime.now() - t0
#             log.info(f"assert create nb{nb}: {tt}")
#
#             def insert(col_w, entities, threads_n):
#                 for _ in range(threads_n):
#                     t1 = datetime.datetime.now()
#                     _, res = col_w.insert(entities)
#                     t2 = datetime.datetime.now() - t1
#                     log.info(f"assert insert round{r}: {t2}")
#
#             # insert
#             threads = []
#             data = cf.gen_default_list_data(nb)[1:3]
#             for r in range(rounds):
#                 for _ in range(threads_num):
#                     t = threading.Thread(target=insert, args=(collection_w, data, threads_num))
#                     threads.append(t)
#                     t.start()
#                 for t in threads:
#                     t.join()
#
#             collection_w.drop()
#             log.info("collection dropped.")
#             sleep(10)
#
