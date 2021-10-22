# import datetime
# import pytest
# from time import sleep
#
# from base.client_base import TestcaseBase
# from common import common_func as cf
# from common import common_type as ct
# from common.common_type import CaseLabel
# from pymilvus import utility
#
# prefix = "e2e_"
# cus_index = {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}, "metric_type": "IP"}
# topK = 5
# cus_search_params = {"params": {"ef": 20}, "metric_type": "IP"}
#
#
# class TestE2e(TestcaseBase):
#     """ Test case of end to end"""
#     @pytest.mark.tags(CaseLabel.L2)
#     # @pytest.mark.parametrize("name", [(cf.gen_unique_str(prefix))])
#     def test_milvus_default(self):
#         from utils.util_log import test_log as log
#         # create
#         name = cf.gen_unique_str(prefix)   # 'load_collection_50m'
#         t0 = datetime.datetime.now()
#         schema = cf.gen_default_collection_schema(auto_id=True)
#         collection_w = self.init_collection_wrap(name=name, schema=schema)
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert create: {tt}")
#         assert collection_w.name == name
#
#         # insert
#         data = cf.gen_default_list_data()[1:3]
#         t0 = datetime.datetime.now()
#         _, res = collection_w.insert(data)
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert insert: {tt}")
#         assert res
#
#         # flush
#         t0 = datetime.datetime.now()
#         n = collection_w.num_entities
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert flush {n}: {tt}")
#
#         # search
#         collection_w.load()
#         search_vectors = cf.gen_vectors(1, ct.default_dim)
#         # search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
#         t0 = datetime.datetime.now()
#         res_1, _ = collection_w.search(data=search_vectors,
#                                        anns_field=ct.default_float_vec_field_name,
#                                        param=cus_search_params,
#                                        limit=1)
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert search: {tt}")
#         assert len(res_1) == 1
#         collection_w.release()
#
#         # index
#         d = cf.gen_default_list_data(nb=5000)[1:3]
#         t0 = datetime.datetime.now()
#         _, res = collection_w.insert(d)
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert insert: {tt}")
#         # assert collection_w.num_entities == len(data[0]) + 5000
#         _index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
#         t0 = datetime.datetime.now()
#         index, _ = collection_w.create_index(field_name=ct.default_float_vec_field_name,
#                                              index_params=cus_index,
#                                              name=cf.gen_unique_str())
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert index: {tt}")
#         assert len(collection_w.indexes) == 1
#         log.debug(f"{utility.index_building_progress(collection_name=collection_w.name)}")
#
#         # search
#         t0 = datetime.datetime.now()
#         collection_w.load()
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert load: {tt}")
#         search_vectors = cf.gen_vectors(10, ct.default_dim)
#         t0 = datetime.datetime.now()
#         res_1, _ = collection_w.search(data=search_vectors,
#                                        anns_field=ct.default_float_vec_field_name,
#                                        param=cus_search_params, limit=topK)
#         tt = datetime.datetime.now() - t0
#         log.debug(f"assert search {len(res_1)}: {tt}")
#         # assert len(res_1) == 1
#         collection_w.release()
#         # collection_w.index()[0].drop()
#
#         # # query
#         # term_expr = f'{ct.default_int64_field_name} in [3001,4001,4999,2999]'
#         # t0 = datetime.datetime.now()
#         # res, _ = collection_w.query(term_expr)
#         # tt = datetime.datetime.now() - t0
#         # log.debug(f"assert query {len(res)}: {tt}")
#         # assert len(res) == 4
#
