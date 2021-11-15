import pytest

from base.client_base import TestcaseBase
from common import common_func as cf
from common import common_type as ct
from utils.util_log import test_log as log
from common.common_type import CaseLabel, CheckTasks

prefix = "delete"
half_nb = ct.default_nb // 2
tmp_nb = 100
tmp_expr = f'{ct.default_int64_field_name} in {[0]}'
query_res_tmp_expr = [{f'{ct.default_int64_field_name}': 0}]
exp_res = "exp_res"


class TestDeleteParams(TestcaseBase):
    """
    Test case of delete interface
    def delete(expr, partition_name=None, timeout=None, **kwargs)
    return MutationResult
    Only the `in` operator is supported in the expr
    """

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize('is_binary', [False, True])
    def test_delete_entities(self, is_binary):
        """
        target: test delete data from collection
        method: 1.create and insert nb with flush
                2.load collection
                3.delete half of nb
                4.query with deleted ids
        expected: Query result is empty
        """
        # init collection with default_nb default data
        collection_w, _, _, ids = self.init_collection_general(prefix, insert_data=True, is_binary=is_binary)[0:4]
        expr = f'{ct.default_int64_field_name} in {ids[:half_nb]}'

        # delete half of data
        del_res = collection_w.delete(expr)[0]
        assert del_res.delete_count == half_nb
        # This flush will not persist the deleted ids, just delay the time to ensure that queryNode consumes deleteMsg
        collection_w.num_entities

        # query with deleted ids
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_without_connection(self):
        """
        target: test delete without connect
        method: delete after remove connection
        expected: raise exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]

        # remove connection and delete
        self.connection_wrap.remove_connection(ct.default_alias)
        res_list, _ = self.connection_wrap.list_connections()
        assert ct.default_alias not in res_list
        error = {ct.err_code: 0, ct.err_msg: "should create connect first"}
        collection_w.delete(expr=tmp_expr, check_task=CheckTasks.err_res, check_items=error)

    # Not Milvus Exception
    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_expr_none(self):
        """
        target: test delete with None expr
        method: delete with None expr
        expected: raise exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        error = {ct.err_code: 0, ct.err_msg: "expr cannot be None"}
        collection_w.delete(expr=None, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("expr", [1, [], ()])
    def test_delete_expr_non_string(self, expr):
        """
        target: test delete with non-string expression
        method: delete with non-string expr
        expected: raise exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        error = {ct.err_code: 0, ct.err_msg: f"expr value {expr} is illegal"}
        collection_w.delete(expr, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("expr", ["12-s", "中文"])
    def test_delete_invalid_expr_string(self, expr):
        """
        target: test delete with invalid string expr
        method: delete with invalid string
        expected: Raise exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        error = {ct.err_code: 1, ct.err_msg: f"failed to create expr plan, expr = {expr}"}
        collection_w.delete(expr, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_expr_empty_value(self):
        """
        target: test delete with empty array expr
        method: delete with expr: "id in []"
        expected: assert num entities
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        expr = f'{ct.default_int64_field_name} in {[]}'

        # delete empty entities
        collection_w.delete(expr)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_expr_single(self):
        """
        target: test delete with one value
        method: delete with expr: "id in [0]"
        expected: Describe num entities by one
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        expr = f'{ct.default_int64_field_name} in {[0]}'
        del_res, _ = collection_w.delete(expr)
        assert del_res.delete_count == 1
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_expr_all_values(self):
        """
        target: test delete with all values
        method: delete with expr: "id in [all]"
        expected: num entities unchanged and deleted data will be not be queried
        """
        # init collection with default_nb default data
        collection_w, _, _, ids = self.init_collection_general(prefix, insert_data=True)[0:4]
        expr = f'{ct.default_int64_field_name} in {ids}'
        del_res, _ = collection_w.delete(expr)

        # assert results
        assert del_res.delete_count == ct.default_nb
        assert collection_w.num_entities == ct.default_nb

        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("ids", [[tmp_nb], [0, tmp_nb]])
    def test_delete_not_existed_values(self, ids):
        """
        target: test delete part/not existed values
        method: delete data not in the collection
        expected: No exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]

        # No exception
        expr = f'{ct.default_int64_field_name} in {ids}'
        res = collection_w.delete(expr=expr)[0]
        # todo assert res.delete_count == 0
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_results,
                           check_items={exp_res: query_res_tmp_expr})

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_expr_inconsistent_values(self):
        """
        target: test delete with inconsistent type values
        method: delete with non-int64 type values
        expected: raise exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        expr = f'{ct.default_int64_field_name} in {[0.0, 1.0]}'

        # Bad exception message
        error = {ct.err_code: 1, ct.err_msg: "failed to create expr plan,"}
        collection_w.delete(expr=expr, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_expr_mix_values(self):
        """
        target: test delete with mix type values
        method: delete with int64 and float values
        expected: raise exception
        """
        # init collection with tmp_nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        expr = f'{ct.default_int64_field_name} in {[0, 1.0]}'

        # Bad exception message
        error = {ct.err_code: 1, ct.err_msg: "failed to create expr plan"}
        collection_w.delete(expr=expr, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L0)
    def test_delete_partition(self):
        """
        target: test delete from partition
        method: delete with partition names
        expected: verify partition entities are deleted
        """
        # init collection and partition
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        partition_w = self.init_partition_wrap(collection_wrap=collection_w)

        # load collection and insert data to partition
        collection_w.load()
        df = cf.gen_default_dataframe_data(tmp_nb)
        partition_w.insert(df)

        # delete ids from partition
        del_res, _ = collection_w.delete(tmp_expr, partition_name=partition_w.name)
        assert del_res.delete_count == 1

        # query with deleted id and query with existed id
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty, partition_names=[partition_w.name])
        res = df.iloc[1:2, :1].to_dict('records')
        collection_w.query(f'{ct.default_int64_field_name} in [1]',
                           check_task=CheckTasks.check_query_results, check_items={exp_res: res})

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_default_partition(self):
        """
        target: test delete from default partition
        method: delete with partition name "_default"
        expected: assert delete successfully
        """
        # create, insert with flush, load collection
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        del_res, _ = collection_w.delete(tmp_expr, partition_name=ct.default_partition_name)
        assert del_res.delete_count == 1
        collection_w.num_entities
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.parametrize("partition_name", [1, [], {}, ()])
    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_non_string_partition_name(self, partition_name):
        """
        target: test delete with non-string partition name
        method: delete with non-string partition name
        expected: Raise exception
        """
        # create, insert with flush, load collection
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]

        error = {ct.err_code: 0, ct.err_msg: f"partition_name value {partition_name} is illegal"}
        collection_w.delete(tmp_expr, partition_name=partition_name, check_task=CheckTasks.err_res, check_items=error)


class TestDeleteOperation(TestcaseBase):
    """
    ******************************************************************
      The following cases are used to test delete interface operations
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_from_empty_collection(self):
        """
        target: test delete entities from an empty collection
        method: create a collection and delete entities
        expected: No exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        collection_w.delete(tmp_expr)[0]
        # todo assert del_res.delete_count == 0

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_entities_repeatedly(self):
        """
        target: test delete entities twice
        method: delete with same expr twice
        expected: No exception for second deletion
        """
        # init collection with nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]

        # assert delete successfully and no exception
        collection_w.delete(expr=tmp_expr)
        collection_w.num_entities
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)
        collection_w.delete(expr=tmp_expr)

    @pytest.mark.xfail(reason="Issue #11641")
    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_after_index(self):
        """
        target: test delete after creating index
        method: 1.insert, flush, load
                2.create index and re-load
                3.delete entities
                4.search
        expected: assert index and deleted id not in search result
        """
        # create collection, insert tmp_nb, flush and load
        collection_w, vectors = self.init_collection_general(prefix, insert_data=True)[0:2]

        # create index
        index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
        collection_w.create_index(ct.default_float_vec_field_name, index_params)
        assert collection_w.has_index()[0]
        collection_w.release()
        collection_w.load()
        # delete entity
        collection_w.delete(tmp_expr)
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)
        assert collection_w.has_index()[0]

        # search with id 0 vectors
        search_res, _ = collection_w.search([vectors[0][ct.default_float_vec_field_name][0]],
                                            ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit)
        assert 0 not in search_res[0].ids

    @pytest.mark.xfail(reason="Issue #11641")
    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_and_index(self):
        """
        target: test delete and create index
        method: 1.insert
                2.delete half
                3.flush and create index
                4.search
        expected: Empty search result
        """
        # init collection and insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data()
        insert_res, _ = collection_w.insert(df)

        # delete half and flush
        expr = f'{ct.default_int64_field_name} in {insert_res.primary_keys[:ct.default_nb // 2]}'
        del_res, _ = collection_w.delete(expr)
        assert collection_w.num_entities == ct.default_nb

        # create index
        index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
        collection_w.create_index(ct.default_float_vec_field_name, index_params)
        assert collection_w.has_index()[0]

        collection_w.load()
        search_res, _ = collection_w.search([df[ct.default_float_vec_field_name][0]],
                                            ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit)
        log.debug(search_res[0].ids)
        # assert search results not contains deleted ids
        inter = set(insert_res.primary_keys[:ct.default_nb // 2]).intersection(set(search_res[0].ids))
        log.debug(inter)
        assert len(inter) == 0

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_query_ids_both_sealed_and_channel(self):
        """
        target: test query that delete ids from both channel and sealed
        method: 1.create and insert
                2.delete id 0 and flush
                3.load and query id 0
                4.insert new id and delete the id
                5.query id 0 and new id
        expected: Empty query result
        """
        # init collection and insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)

        # delete id 0 and flush
        del_res, _ = collection_w.delete(tmp_expr)
        assert del_res.delete_count == 1
        assert collection_w.num_entities == tmp_nb

        # load and query id 0
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

        # insert id tmp_nb and delete id 0 and tmp_nb
        df_new = cf.gen_default_dataframe_data(nb=1, start=tmp_nb)
        collection_w.insert(df_new)
        collection_w.delete(expr=f'{ct.default_int64_field_name} in {[tmp_nb]}')

        # query with id 0 and tmp_nb
        collection_w.query(expr=f'{ct.default_int64_field_name} in {[0, tmp_nb]}',
                           check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_search(self):
        """
        target: test delete and search
        method: search entities after it was deleted
        expected: deleted entity is not in the search result
        """
        # init collection with nb default data
        collection_w, _, _, ids = self.init_collection_general(prefix, insert_data=True)[0:4]
        entity, _ = collection_w.query(tmp_expr, output_fields=["%"])
        search_res, _ = collection_w.search([entity[0][ct.default_float_vec_field_name]],
                                            ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit)
        # assert search results contains entity
        assert 0 in search_res[0].ids

        expr = f'{ct.default_int64_field_name} in {ids[:ct.default_nb // 2]}'
        collection_w.delete(expr)
        search_res_2, _ = collection_w.search([entity[0][ct.default_float_vec_field_name]],
                                              ct.default_float_vec_field_name,
                                              ct.default_search_params, ct.default_limit)
        # assert search result is not equal to entity
        log.debug(f"Second search result ids: {search_res_2[0].ids}")
        inter = set(ids[:ct.default_nb // 2]).intersection(set(search_res_2[0].ids))
        assert len(inter) == 0

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_expr_repeated_values(self):
        """
        target: test delete with repeated values
        method: 1.insert data with unique primary keys
                2.delete with repeated values: 'id in [0, 0]'
        expected: delete one entity
        """
        # init collection with nb default data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]
        expr = f'{ct.default_int64_field_name} in {[0, 0, 0]}'
        del_res, _ = collection_w.delete(expr)
        assert del_res.delete_count == 3
        collection_w.num_entities
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_duplicate_primary_keys(self):
        """
        target: test delete from duplicate primary keys
        method: 1.insert data with dup ids
                2.delete with repeated or not values
        expected: currently only delete one entity, query get one entity
        todo delete all entities
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(nb=tmp_nb)
        df[ct.default_int64_field_name] = 0
        collection_w.insert(df)
        assert collection_w.num_entities == tmp_nb
        del_res, _ = collection_w.delete(tmp_expr)
        collection_w.load()

        # Just one query res and search res, because de-dup
        res, _ = collection_w.query(tmp_expr, output_fields=["*"])
        assert res[0][ct.default_float_field_name] != 0.0

        search_res, _ = collection_w.search([df[ct.default_float_vec_field_name][0]],
                                            ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit, output_fields=[ct.default_int64_field_name, ct.default_float_field_name])
        assert search_res[0][0].entity._row_data[ct.default_float_field_name] != 0.0

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_empty_partition(self):
        """
        target: test delete empty partition
        method: delete from an empty partition
        expected: No exception
        """
        # init collection and partition
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        partition_w = self.init_partition_wrap(collection_wrap=collection_w)

        collection_w.delete(tmp_expr, partition_name=partition_w.name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_not_existed_partition(self):
        """
        target: test delete from an not existed partition
        method: delete from an fake partition
        expected: raise exception
        """
        # init collection with tmp_nb data
        collection_w = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True)[0]

        # raise exception
        error = {ct.err_code: 1, ct.err_msg: f"partitionID of partitionName:{ct.default_tag} can not be find"}
        collection_w.delete(tmp_expr, partition_name=ct.default_tag, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_from_partition_with_another_ids(self):
        """
        target: delete another partition entities from partition
        method: 1.insert nb entities into two partitions in half
                2.delete entities from partition_1 with partition_2 values
                3.delete entities from partition_1 with partition_1 values
        expected: Entities in partition_1 will be deleted
        """
        half = tmp_nb // 2
        # create, insert, flush, load
        collection_w, partition_w, _, _ = self.insert_entities_into_two_partitions_in_half(half)

        # delete entities from another partition
        expr = f'{ct.default_int64_field_name} in {[0]}'
        collection_w.delete(expr, partition_name=ct.default_partition_name)
        collection_w.query(expr, check_task=CheckTasks.check_query_results, check_items={exp_res: query_res_tmp_expr})

        # delete entities from own partition
        collection_w.delete(expr, partition_name=partition_w.name)
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_from_partitions_with_same_ids(self):
        """
        target: test delete same ids from two partitions with same data
        method: 1.insert same nb data into two partitions
                2.delete same ids from partition_1
        expected: The data only in partition_1 will be deleted
        """
        # init collection and partition
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        partition_w = self.init_partition_wrap(collection_wrap=collection_w)

        # insert same data into partition_w and default partition
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        partition_w.insert(df)

        # delete same id 0 from default_partition, and query on it get empty result
        collection_w.delete(tmp_expr, partition_name=ct.default_partition_name)
        assert collection_w.num_entities == tmp_nb * 2
        collection_w.load()
        collection_w.query(tmp_expr, partition_names=[ct.default_partition_name],
                           check_task=CheckTasks.check_query_empty)

        # query on partition_w with id 0 and get an result
        collection_w.query(tmp_expr, partition_names=[partition_w.name],
                           check_task=CheckTasks.check_query_results, check_items={exp_res: query_res_tmp_expr})

    @pytest.mark.tags(CaseLabel.L0)
    def test_delete_auto_id_collection(self):
        """
        target: test delete from auto_id collection
        method: delete entities from auto_id=true collection
        expected: versify delete successfully
        """
        # init an auto_id collection and insert tmp_nb data
        collection_w, _, _, ids = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True, auto_id=True)[0:4]

        # delete with insert ids
        expr = f'{ct.default_int64_field_name} in {[ids[0]]}'
        res, _ = collection_w.delete(expr)

        # verify delete result
        assert res.delete_count == 1
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_query_without_loading(self):
        """
        target: test delete and query without loading
        method: 1.insert and flush data
                2.delete ids
                3.query without loading
        expected: Raise exception
        """
        # create collection, insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        assert collection_w.num_entities == tmp_nb

        # delete
        res = collection_w.delete(tmp_expr)[0]
        assert res.delete_count == 1

        # query without loading and raise exception
        error = {ct.err_code: 1, ct.err_msg: f"collection {collection_w.name} was not loaded into memory"}
        collection_w.query(expr=tmp_expr, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_sealed_segment_without_flush(self):
        """
        target: test delete without flush
        method: 1.insert and flush data
                2.delete ids from collection and no flush
                3.load and query with id
        expected: No query result
        """
        # create collection, insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        assert collection_w.num_entities == tmp_nb

        # delete
        del_res, _ = collection_w.delete(tmp_expr)
        assert del_res.delete_count == 1

        # load and query with id
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_growing_data_channel_delete(self):
        """
        target: test delete entities from growing segment, and channel deleteMsg
        method: 1.create collection
                2.load collection
                3.insert data and delete ids
                4.query deleted ids
        expected: No query result
        """
        # create collection
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        # load collection and the queryNode watch the insertChannel
        collection_w.load()
        # insert data
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        # delete id 0
        del_res = collection_w.delete(tmp_expr)[0]
        assert del_res.delete_count == 1
        # query id 0
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_sealed_data_channel_delete(self):
        """
        target: test delete sealed data and get deleteMsg from insertChannel
        method: 1.create, insert and flush data
                2.load collection
                3.delete id without flush
                4.query deleted ids (queryNode get deleted ids from channel not persistence)
        expected: Delete successfully and no query result
        """
        # create collection and insert flush data
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        assert collection_w.num_entities == tmp_nb

        # load collection and queryNode subscribe channel
        collection_w.load()

        # delete ids and query
        collection_w.delete(tmp_expr)
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_sealed_segment_with_flush(self):
        """
        target: test delete data from sealed segment and flush delta log
        method: 1.create and insert and flush data
                2.delete entities and flush (insert and flush)
                3.load collection (load data and delta log)
                4.query deleted ids
        expected: No query result
        """
        # create collection
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        # insert and flush data
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        # assert collection_w.num_entities == tmp_nb

        # delete id 0
        del_res = collection_w.delete(tmp_expr)[0]
        assert del_res.delete_count == 1

        # insert data and flush data and delete ids.
        # if no insert, datanode will not really flush delete ids
        collection_w.insert(cf.gen_default_dataframe_data(nb=1, start=tmp_nb))
        log.info(f'Collection num entities: {collection_w.num_entities}')

        # load and query id 0
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_sealed_segment_with_twice_flush(self):
        """
        target: test delete data from sealed segment and flush delta log
        method: 1.create and insert and flush data
                2.delete entities and flush (insert and flush)
                3.load collection (load data and delta log)
                4.query deleted ids
        expected: No query result
        """
        # create collection
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        # insert and flush data
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)
        assert collection_w.num_entities == tmp_nb

        # delete id 0 and flush
        del_res = collection_w.delete(tmp_expr)[0]
        assert del_res.delete_count == 1
        collection_w.insert(cf.gen_default_dataframe_data(nb=1, start=tmp_nb))
        log.info(collection_w.num_entities)
        # load and query id 0
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L1)
    def test_delete_sealed_data_sealed_delete(self):
        """
        target: test delete with sealed data and sealed delete request
        method: 1.create, insert
                2.delete and flush (will flush data and delete)
                3.load and query
        expected: Empty query result
        """
        # create collection
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        # insert without flush
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)

        # delete id 0 and flush
        del_res = collection_w.delete(tmp_expr)[0]
        assert del_res.delete_count == 1
        assert collection_w.num_entities == tmp_nb

        # load and query id 0
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_insert_same_entity(self):
        """
        target: test delete and insert same entity
        method: 1.delete entity one
                2.insert entity one
                3.query  entity one
        expected: verify query result
        """
        # init collection and insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        collection_w.load()
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)

        # delete
        del_res, _ = collection_w.delete(tmp_expr)
        assert del_res.delete_count == 1
        # assert collection_w.num_entities == tmp_nb
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

        # insert entity with primary key 0
        collection_w.insert(df[:1])

        # query entity one
        res = df.iloc[0:1, :1].to_dict('records')
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_results, check_items={'exp_res': res})

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_entity_loop(self):
        """
        target: test delete all entities one by one in a loop
        method: delete data one by one for a loop
        expected: No exception
        """
        # init an auto_id collection and insert tmp_nb data, flush and load
        collection_w, _, _, ids = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True, auto_id=True)[0:4]

        for del_id in ids:
            expr = f'{ct.default_int64_field_name} in {[del_id]}'
            res = collection_w.delete(expr)[0]
            assert res.delete_count == 1

        # query with all ids
        expr = f'{ct.default_int64_field_name} in {ids}'
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_flush_loop(self):
        """
        target: test delete and flush in a loop
        method: in a loop, delete batch and flush, until delete all entities
        expected: No exception
        """
        # init an auto_id collection and insert tmp_nb data
        collection_w, _, _, ids = self.init_collection_general(prefix, nb=tmp_nb, insert_data=True, auto_id=True)[0:4]

        batch = 10
        for i in range(tmp_nb // batch):
            expr = f'{ct.default_int64_field_name} in {ids[i * batch: (i + 1) * batch]}'
            res, _ = collection_w.delete(expr)
            assert res.delete_count == batch
            assert collection_w.num_entities == tmp_nb

        # query with all ids
        expr = f'{ct.default_int64_field_name} in {ids}'
        collection_w.query(expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_merge_same_id_channel_and_sealed(self):
        """
        target: test merge same delete ids from channel and sealed
        method: 1.create, insert
                2.delete id and flush (data and deleted become sealed)
                3.load and query (verify delete successfully)
                4.insert entity with deleted id
                5.delete id
                6.query with id
        expected: Empty query result
        """
        # init collection and insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix), shards_num=1)
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)

        # delete id 0 and flush
        del_res, _ = collection_w.delete(tmp_expr)
        assert del_res.delete_count == 1
        assert collection_w.num_entities == tmp_nb

        # load and query id 0
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

        # re-insert id 0 and re-delete id 0
        collection_w.insert(df[:1])
        collection_w.delete(tmp_expr)
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    def test_delete_merge_ids_channel_and_sealed(self):
        """
        target: test merge deleted ids come from both channel and sealed
        method: 1.create, insert ids [0, tmp_nb) with shard_num=1
                2.delete id 0 and flush
                3.load and query with id 0
                4.delete id 1 (merge same segment deleted ids 0 and 1)
                5.query with id 0 and 1
        expected: Empty query result
        """
        # init collection and insert data without flush
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix), shards_num=1)
        df = cf.gen_default_dataframe_data(tmp_nb)
        collection_w.insert(df)

        # delete id 0 and flush
        del_res, _ = collection_w.delete(tmp_expr)
        assert del_res.delete_count == 1
        assert collection_w.num_entities == tmp_nb

        # load and query id 0
        collection_w.load()
        collection_w.query(tmp_expr, check_task=CheckTasks.check_query_empty)

        # delete id 1 and query id 0 and 1
        collection_w.delete(expr=f'{ct.default_int64_field_name} in {[1]}')
        collection_w.query(expr=f'{ct.default_int64_field_name} in {[0, 1]}', check_task=CheckTasks.check_query_empty)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="TODO")
    def test_delete_multi_threading(self):
        """
        target: test delete multi threading
        method: delete multi threading
        expected: delete successfully
        """
        pass
