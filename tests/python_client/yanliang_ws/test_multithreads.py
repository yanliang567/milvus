# import datetime
# import time
#
# import pytest
# import threading
#
# from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
# from base.client_base import TestcaseBase
# from common import common_func as cf
# from common import common_type as ct
# from common.common_type import CaseLabel
# from utils.util_log import test_log as log
#
# prefix = "e2e_"
#
#
# class TestE2e(TestcaseBase):
#     """ Test case of end to end"""
#     @pytest.mark.tags(CaseLabel.L3)
#     # @pytest.mark.timeout(60)
#     # @pytest.mark.parametrize("name", [(cf.gen_unique_str(prefix))])
#     def test_milvus_qps_max(self):
#         # create
#         name = 'e2e__dkooNG7R'  # cf.gen_unique_str(prefix)
#         t0 = datetime.datetime.now()
#         collection_w = self.init_collection_wrap(name=name)
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert create: {tt}")
#         assert collection_w.name == name
#
#         # search
#         collection_w.load()
#         search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
#
#         def collection_search(collection, nums):
#             fails = 0
#             latency = 0.5
#             while fails < 3:
#                 _search_vectors = cf.gen_vectors(1, ct.default_dim)
#                 t1 = time.time()
#                 collection.search(data=_search_vectors,
#                                   anns_field=ct.default_float_vec_field_name,
#                                   param=search_params,
#                                   limit=1)
#                 t2 = time.time() - t1
#                 log.debug(f"info  thread{nums} search: {t2}")
#                 if t2 >= latency:
#                     fails += 1
#                 else:
#                     fails = 0
#             return -1
#
#         rounds = 0
#         with ThreadPoolExecutor(max_workers=100) as executor:
#             all_tasks = []
#             while True:
#                 rounds += 1
#                 log.debug(f"rounds: {rounds}")
#                 task = executor.submit(collection_search,collection_w.collection,rounds)
#                 all_tasks.append(task)
#                 searcher_res = wait(all_tasks, return_when=FIRST_COMPLETED)
#                 log.debug(f"searcher_res: {searcher_res}")
#                 break;
#             wait(all_tasks, timeout=0.1)
#
#
#
#                 else:
#                     executor.submit(collection_search,
#                                     collection_w.collection,
#                                     False, rounds)
#             executor.shutdown(wait=False)
#             executor.
#             log.debug(f"search latency timeout with rounds{rounds}")
#             log.debug(f"searcher_res: {searcher_res}")
#         log.debug("completed")
#
#
