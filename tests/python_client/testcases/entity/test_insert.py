import copy
import threading

import pytest
from pymilvus import DataType, ParamError, BaseException
from utils import util_pymilvus as ut
from common.constants import default_entity, default_entities, default_binary_entity, default_binary_entities, \
    default_fields
from common.common_type import CaseLabel
from utils.util_log import test_log as log

# ADD_TIMEOUT = 60
uid = "test_insert"
field_name = ut.default_float_vec_field_name
binary_field_name = ut.default_binary_vec_field_name
default_nb = ut.default_nb
row_count = ut.row_count
default_tag = ut.default_tag
default_single_query = {
            "data": ut.gen_vectors(1, ut.default_dim),
            "anns_field": ut.default_float_vec_field_name,
            "param": {"metric_type": "L2", "params": {"nprobe": 10}},
            "limit": 10,
        }


class TestInsertBase:
    """
    ******************************************************************
      The following cases are used to test `insert` function
    ******************************************************************
    """

    @pytest.fixture(
        scope="function",
        params=ut.gen_simple_index()
    )
    def get_simple_index(self, request, connect):
        if request.param["index_type"] in ut.index_cpu_not_support():
            pytest.skip("CPU not support index_type: ivf_sq8h")
        log.info(request.param)
        return request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_single_filter_fields()
    )
    def get_filter_field(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_single_vector_fields()
    )
    def get_vector_field(self, request):
        yield request.param

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_collection_not_existed(self, connect):
        """
        target: test insert, with collection not existed
        method: insert entity into a random named collection
        expected: raise a BaseException
        """
        collection_name = ut.gen_unique_str(uid)
        with pytest.raises(BaseException) as e:
            connect.insert(collection_name, default_entities)

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_search(self, connect, collection):
        """
        target: test search entity after insert entity after a while
        method: insert entity, sleep, and search collection
        expected: no error raised
        """
        result = connect.insert(collection, default_entities)
        connect.flush([collection])
        connect.load_collection(collection)
        res = connect.search(collection, **default_single_query)
        assert len(res[0]) == ut.default_top_k

    @pytest.mark.tags(CaseLabel.L2)
    def _test_insert_segment_row_count(self, connect, collection):
        nb = ut.default_segment_row_limit + 1
        result = connect.insert(collection, ut.gen_entities(nb))
        connect.flush([collection])
        assert len(result.primary_keys) == nb
        stats = connect.get_collection_stats(collection)
        assert len(stats['partitions'][0]['segments']) == 2
        for segment in stats['partitions'][0]['segments']:
            assert segment['row_count'] in [ut.default_segment_row_limit, 1]

    @pytest.fixture(
        scope="function",
        params=[
            1,
            2000
        ],
    )
    def insert_count(self, request):
        yield request.param

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_ids(self, connect, id_collection, insert_count):
        """
        target: test insert entities in collection, use customize ids
        method: create collection and insert entities in it, check the ids returned and
                the collection length after entities inserted
        expected: the length of ids and the collection row count
        """
        nb = insert_count
        ids = [i for i in range(nb)]
        entities = ut.gen_entities(nb)
        entities[0]["values"] = ids
        result = connect.insert(id_collection, entities)
        connect.flush([id_collection])
        assert len(result.primary_keys) == nb
        assert result.primary_keys == ids
        stats = connect.get_collection_stats(id_collection)
        assert stats[row_count] == nb

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_ids_fields(self, connect, get_filter_field, get_vector_field):
        """
        target: test create normal collection with different fields, insert entities into id with ids
        method: create collection with diff fields: metric/field_type/..., insert, and count
        expected: row count correct
        """
        nb = 5
        filter_field = get_filter_field
        vector_field = get_vector_field
        collection_name = ut.gen_unique_str("test_collection")
        fields = {
            "fields": [ut.gen_primary_field(), filter_field, vector_field],
            "auto_id": False
        }
        connect.create_collection(collection_name, fields)
        ids = [i for i in range(nb)]
        entities = ut.gen_entities_by_fields(fields["fields"], nb, ut.default_dim, ids)
        log.info(entities)
        result = connect.insert(collection_name, entities)
        assert result.primary_keys == ids
        connect.flush([collection_name])
        stats = connect.get_collection_stats(collection_name)
        assert stats[row_count] == nb

    @pytest.mark.tags(CaseLabel.L2)
    # @pytest.mark.timeout(ADD_TIMEOUT)
    def test_insert_ids_not_match(self, connect, id_collection, insert_count):
        """
        target: test insert entities in collection without ids
        method: create id_collection and insert entities without
        expected: exception raised
        """
        nb = insert_count
        with pytest.raises(Exception) as e:
            entities = ut.gen_entities(nb)
            del entities[0]
            connect.insert(id_collection, entities)

    # TODO
    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_twice_ids_no_ids(self, connect, id_collection):
        """
        target: check the result of insert, with params ids and no ids
        method: test insert vectors twice, use customize ids first, and then use no ids
        expected:  BaseException raised
        """
        ids = [i for i in range(default_nb)]
        entities = copy.deepcopy(default_entities)
        entities[0]["values"] = ids
        connect.insert(id_collection, entities)
        with pytest.raises(Exception) as e:
            del entities[0]
            connect.insert(id_collection, entities)

    @pytest.mark.tags(CaseLabel.L2)
    # @pytest.mark.timeout(ADD_TIMEOUT)
    def test_insert_not_ids(self, connect, id_collection):
        """
        target: check the result of insert, with params ids and no ids
        method: test insert vectors twice, use not ids first, and then use customize ids
        expected:  error raised
        """
        entities = copy.deepcopy(default_entities)
        del entities[0]
        with pytest.raises(Exception) as e:
            connect.insert(id_collection, entities)

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_ids_length_not_match_batch(self, connect, id_collection):
        """
        target: test insert vectors in collection, use customize ids, len(ids) != len(vectors)
        method: create collection and insert vectors in it
        expected: raise exception
        """
        ids = [i for i in range(1, default_nb)]
        log.info(len(ids))
        entities = copy.deepcopy(default_entities)
        entities[0]["values"] = ids
        with pytest.raises(Exception) as e:
            connect.insert(id_collection, entities)

    @pytest.mark.tags(CaseLabel.L2)
    # @pytest.mark.timeout(ADD_TIMEOUT)
    def test_insert_ids_length_not_match_single(self, connect, id_collection):
        """
        target: test insert vectors in collection, use customize ids, len(ids) != len(vectors)
        method: create collection and insert vectors in it
        expected: raise exception
        """
        ids = [i for i in range(1, default_nb)]
        log.info(len(ids))
        entity = copy.deepcopy(default_entity)
        entity[0]["values"] = ids
        with pytest.raises(Exception) as e:
            connect.insert(id_collection, entity)

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_partition(self, connect, collection):
        """
        target: test insert entities in collection created before
        method: create collection and insert entities in it, with the partition_name param
        expected: the collection row count equals to nq
        """
        connect.create_partition(collection, default_tag)
        result = connect.insert(collection, default_entities, partition_name=default_tag)
        assert len(result.primary_keys) == default_nb
        assert connect.has_partition(collection, default_tag)
        connect.flush([collection])
        stats = connect.get_collection_stats(collection)
        assert stats[row_count] == default_nb

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_partition_with_ids(self, connect, id_collection):
        """
        target: test insert entities in collection created before, insert with ids
        method: create collection and insert entities in it, with the partition_name param
        expected: the collection row count equals to nq
        """
        connect.create_partition(id_collection, default_tag)
        ids = [i for i in range(default_nb)]
        entities = ut.gen_entities(default_nb)
        entities[0]["values"] = ids
        result = connect.insert(id_collection, entities, partition_name=default_tag)
        assert result.primary_keys == ids
        log.info(connect.describe_collection(id_collection))

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_default_partition(self, connect, collection):
        """
        target: test insert entities into default partition
        method: create partition and insert info collection without tag params
        expected: the collection row count equals to nb
        """
        result = connect.insert(collection, default_entities, partition_name=ut.default_partition_name)
        assert len(result.primary_keys) == default_nb
        connect.flush([collection])
        stats = connect.get_collection_stats(collection)
        assert stats[row_count] == default_nb

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_partition_not_existed(self, connect, collection):
        """
        target: test insert entities in collection created before
        method: create collection and insert entities in it, with the not existed partition_name param
        expected: error raised
        """
        tag = ut.gen_unique_str()
        with pytest.raises(Exception) as e:
            connect.insert(collection, default_entities, partition_name=tag)

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_partition_repeatedly(self, connect, collection):
        """
        target: test insert entities in collection created before
        method: create collection and insert entities in it repeatedly, with the partition_name param
        expected: the collection row count equals to nq
        """
        connect.create_partition(collection, default_tag)
        result = connect.insert(collection, default_entities, partition_name=default_tag)
        result = connect.insert(collection, default_entities, partition_name=default_tag)
        connect.flush([collection])
        res = connect.get_collection_stats(collection)
        assert res[row_count] == 2 * default_nb

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_field_type_not_match(self, connect, collection):
        """
        target: test insert entities, with the entity field type updated
        method: update entity field type
        expected: error raised
        """
        tmp_entity = ut.update_field_type(copy.deepcopy(default_entity), "int64", DataType.FLOAT)
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_field_vector_more(self, connect, collection):
        """
        target: test insert entities, with more fields than collection schema
        method: add entity vector field
        expected: error raised
        """
        tmp_entity = ut.add_vector_field(default_nb, ut.default_dim)
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_no_field_vector_value(self, connect, collection):
        """
        target: test insert entities, with no vector field value
        method: remove entity values of vector field
        expected: error raised
        """
        tmp_entity = copy.deepcopy(default_entity)
        del tmp_entity[-1]["values"]
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_no_field_vector_type(self, connect, collection):
        """
        target: test insert entities, with no vector field type
        method: remove entity vector field
        expected: error raised
        """
        tmp_entity = copy.deepcopy(default_entity)
        del tmp_entity[-1]["type"]
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_no_field_vector_name(self, connect, collection):
        """
        target: test insert entities, with no vector field name
        method: remove entity vector field
        expected: error raised
        """
        tmp_entity = copy.deepcopy(default_entity)
        del tmp_entity[-1]["name"]
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.xfail(reason="issue 15416")
    # @pytest.mark.timeout(30)
    def test_collection_insert_rows_count_multi_threading(self, args, collection):
        """
        target: test collection rows_count is correct or not with multi threading
        method: create collection and insert entities in it(idmap),
                assert the value returned by count_entities method is equal to length of entities
        expected: the count is equal to the length of entities
        """
        if args["handler"] == "HTTP":
            pytest.skip("Skip test in http mode")
        thread_num = 8
        threads = []
        milvus = ut.get_milvus(host=args["ip"], port=args["port"], handler=args["handler"], try_connect=False)

        def insert(thread_i):
            log.info("In thread-%d" % thread_i)
            result = milvus.insert(collection, default_entities)
            milvus.flush([collection])

        for i in range(thread_num):
            x = threading.Thread(target=insert, args=(i,))
            threads.append(x)
            x.start()
        for th in threads:
            th.join()
        stats = milvus.get_collection_stats(collection)
        assert stats[row_count] == thread_num * default_nb

    # TODO: unable to set config
    @pytest.mark.tags(CaseLabel.L2)
    def _test_insert_disable_auto_flush(self, connect, collection):
        """
        target: test insert entities, with disable auto-flush
        method: disable auto-flush and insert, get entity
        expected: the count is equal to 0
        """
        delete_nums = 500
        ut.disable_flush(connect)
        result = connect.insert(collection, default_entities)
        ids = result.primary_keys
        res = connect.get_entity_by_id(collection, ids[:delete_nums])
        assert len(res) == delete_nums
        assert res[0] is None


class TestInsertBinary:
    @pytest.fixture(
        scope="function",
        params=ut.gen_binary_index()
    )
    def get_binary_index(self, request):
        request.param["metric_type"] = "JACCARD"
        return request.param

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_binary_partition(self, connect, binary_collection):
        """
        target: test insert entities and create partition tag
        method: create collection and insert binary entities in it, with the partition_name param
        expected: the collection row count equals to nb
        """
        connect.create_partition(binary_collection, default_tag)
        result = connect.insert(binary_collection, default_binary_entities, partition_name=default_tag)
        assert len(result.primary_keys) == default_nb
        assert connect.has_partition(binary_collection, default_tag)
        connect.flush([binary_collection])
        stats = connect.get_collection_stats(binary_collection)
        assert stats[row_count] == default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_binary_multi_times(self, connect, binary_collection):
        """
        target: test insert entities multi times and final flush
        method: create collection and insert binary entity multi and final flush
        expected: the collection row count equals to nb
        """
        for i in range(default_nb):
            result = connect.insert(binary_collection, default_binary_entity)
            assert len(result.primary_keys) == 1
        connect.flush([binary_collection])
        stats = connect.get_collection_stats(binary_collection)
        assert stats[row_count] == default_nb

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_binary_after_create_index(self, connect, binary_collection, get_binary_index):
        """
        target: test insert binary entities after build index
        method: build index and insert entities
        expected: no error raised
        """
        connect.create_index(binary_collection, binary_field_name, get_binary_index)
        result = connect.insert(binary_collection, default_binary_entities)
        assert len(result.primary_keys) == default_nb
        connect.flush([binary_collection])
        index = connect.describe_index(binary_collection, "")
        ut.create_target_index(get_binary_index, binary_field_name)
        assert index == get_binary_index

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_binary_create_index(self, connect, binary_collection, get_binary_index):
        """
        target: test build index insert after vector
        method: insert vector and build index
        expected: no error raised
        """
        result = connect.insert(binary_collection, default_binary_entities)
        assert len(result.primary_keys) == default_nb
        connect.flush([binary_collection])
        connect.create_index(binary_collection, binary_field_name, get_binary_index)
        index = connect.describe_index(binary_collection, "")
        ut.create_target_index(get_binary_index, binary_field_name)
        assert index == get_binary_index

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_binary_search(self, connect, binary_collection):
        """
        target: test search vector after insert vector after a while
        method: insert vector, sleep, and search collection
        expected: no error raised
        """
        result = connect.insert(binary_collection, default_binary_entities)
        connect.flush([binary_collection])
        query, _ = ut.gen_search_vectors_params(binary_field_name, default_binary_entities,
                                                ut.default_top_k, 1, metric_type="JACCARD")
        connect.load_collection(binary_collection)
        res = connect.search(binary_collection, **query)
        log.debug(res)
        assert len(res[0]) == ut.default_top_k


class TestInsertAsync:
    @pytest.fixture(scope="function", autouse=True)
    def skip_http_check(self, args):
        if args["handler"] == "HTTP":
            pytest.skip("skip in http mode")

    @pytest.fixture(
        scope="function",
        params=[
            1,
            1000
        ],
    )
    def insert_count(self, request):
        yield request.param

    def check_status(self, result):
        log.info("In callback check status")
        assert not result

    def check_result(self, result):
        log.info("In callback check results")
        assert result

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_invalid_params(self, connect):
        """
        target: test insert vectors with different length of vectors
        method: set different vectors as insert method params
        expected: length of ids is equal to the length of vectors
        """
        collection_new = ut.gen_unique_str()
        future = connect.insert(collection_new, default_entities, _async=True)
        future.done()
        with pytest.raises(Exception) as e:
            result = future.result()

    # 1339
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_invalid_params_raise_exception(self, connect, collection):
        """
        target: test insert vectors with different length of vectors
        method: set different vectors as insert method params
        expected: length of ids is equal to the length of vectors
        """
        entities = []
        future = connect.insert(collection, entities, _async=True)
        future.done()
        with pytest.raises(Exception) as e:
            future.result()


class TestInsertMultiCollections:
    """
    ******************************************************************
      The following cases are used to test `insert` function
    ******************************************************************
    """

    @pytest.fixture(
        scope="function",
        params=ut.gen_simple_index()
    )
    def get_simple_index(self, request, connect):
        log.info(request.param)
        # if str(connect._cmd("mode")) == "CPU":
        #     if request.param["index_type"] in index_cpu_not_support():
        #         pytest.skip("sq8h not support in CPU mode")
        return request.param

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_entity_multi_collections(self, connect):
        """
        target: test insert entities
        method: create 10 collections and insert entities into them in turn
        expected: row count
        """
        collection_num = 10
        collection_list = []
        for i in range(collection_num):
            collection_name = ut.gen_unique_str(uid)
            collection_list.append(collection_name)
            connect.create_collection(collection_name, default_fields)
            result = connect.insert(collection_name, default_entities)
            connect.flush([collection_name])
            assert len(result.primary_keys) == default_nb
            stats = connect.get_collection_stats(collection_name)
            assert stats[row_count] == default_nb
        for i in range(collection_num):
            connect.drop_collection(collection_list[i])

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_drop_collection_insert_entity_another(self, connect, collection):
        """
        target: test insert vector to collection_1 after collection_2 deleted
        method: delete collection_2 and insert vector to collection_1
        expected: row count equals the length of entities inserted
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        connect.drop_collection(collection)
        result = connect.insert(collection_name, default_entity)
        connect.flush([collection_name])
        assert len(result.primary_keys) == 1

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_create_index_insert_entity_another(self, connect, collection, get_simple_index):
        """
        target: test insert vector to collection_2 after build index for collection_1
        method: build index and insert vector
        expected: status ok
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        connect.create_index(collection, field_name, get_simple_index)
        result = connect.insert(collection_name, default_entity)
        assert len(result.primary_keys) == 1
        if get_simple_index["index_type"] != "FLAT":
            index = connect.describe_index(collection, "")
            ut.create_target_index(get_simple_index, field_name)
            assert index == get_simple_index
        connect.drop_collection(collection_name)

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_entity_create_index_another(self, connect, collection, get_simple_index):
        """
        target: test insert vector to collection_2 after build index for collection_1
        method: build index and insert vector
        expected: status ok
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        result = connect.insert(collection, default_entity)
        connect.flush([collection])
        connect.create_index(collection_name, field_name, get_simple_index)
        if get_simple_index["index_type"] != "FLAT":
            index = connect.describe_index(collection_name, "")
            ut.create_target_index(get_simple_index, field_name)
            assert index == get_simple_index
        stats = connect.get_collection_stats(collection)
        assert stats[row_count] == 1

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_entity_sleep_create_index_another(self, connect, collection, get_simple_index):
        """
        target: test insert vector to collection_2 after build index for collection_1 for a while
        method: build index and insert vector
        expected: status ok
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        result = connect.insert(collection, default_entity)
        connect.flush([collection])
        connect.create_index(collection_name, field_name, get_simple_index)
        stats = connect.get_collection_stats(collection)
        assert stats[row_count] == 1

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L1)
    def test_search_entity_insert_entity_another(self, connect, collection):
        """
        target: test insert entity to collection_1 after search collection_2
        method: search collection and insert entity
        expected: status ok
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        connect.load_collection(collection)
        res = connect.search(collection, **default_single_query)
        assert len(res[0]) == 0
        connect.insert(collection_name, default_entity)
        connect.flush([collection_name])
        stats = connect.get_collection_stats(collection_name)
        assert stats[row_count] == 1

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_entity_search_entity_another(self, connect, collection):
        """
        target: test insert entity to collection_1 after search collection_2
        method: search collection and insert entity
        expected: status ok
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        result = connect.insert(collection, default_entity)
        connect.flush([collection])
        connect.load_collection(collection_name)
        res = connect.search(collection_name, **default_single_query)
        stats = connect.get_collection_stats(collection)
        assert stats[row_count] == 1

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_entity_sleep_search_entity_another(self, connect, collection):
        """
        target: test insert entity to collection_1 after search collection_2 a while
        method: search collection, sleep, and insert entity
        expected: status ok
        """
        collection_name = ut.gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        result = connect.insert(collection, default_entity)
        connect.flush([collection])
        connect.load_collection(collection_name)
        res = connect.search(collection_name, **default_single_query)
        assert len(res[0]) == 0

    # @pytest.mark.timeout(ADD_TIMEOUT)
    @pytest.mark.tags(CaseLabel.L2)
    def _test_insert_entity_during_release_collection(self, connect, collection):
        """
        target: test insert entity during release
        method: release collection async, then do insert operation
        expected: insert ok
        """
        for i in range(10):
            connect.insert(collection, default_entities)
        connect.flush([collection])
        connect.load_collection(collection)

        def release():
            connect.release_collection(collection)

        t = threading.Thread(target=release, args=(collection,))
        t.start()
        result = connect.insert(collection, default_entities)
        assert len(result.primary_keys) == default_nb


class TestInsertInvalid(object):
    """
    Test inserting vectors with invalid collection names
    """

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_collection_name(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_tag_name(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_field_name(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_field_type(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_field_int_value(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_ints()
    )
    def get_entity_id(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_vectors()
    )
    def get_field_vectors_value(self, request):
        yield request.param

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_ids_invalid(self, connect, id_collection, get_entity_id):
        """
        target: test insert, with using customize ids, which are not int64
        method: create collection and insert entities in it
        expected: raise exception
        """
        entity_id = get_entity_id
        ids = [entity_id for _ in range(default_nb)]
        with pytest.raises(Exception):
            connect.insert(id_collection, default_entities, ids)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_collection_name(self, connect, get_collection_name):
        """
        target: test insert with invalid scenario
        method: insert with invalid collection name
        expected: raise exception
        """
        collection_name = get_collection_name
        with pytest.raises(Exception):
            connect.insert(collection_name, default_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_partition_name(self, connect, collection, get_tag_name):
        """
        target: test insert with invalid scenario
        method: insert with invalid partition name
        expected: raise exception
        """
        tag_name = get_tag_name
        connect.create_partition(collection, default_tag)
        if tag_name is not None:
            with pytest.raises(Exception):
                connect.insert(collection, default_entity, partition_name=tag_name)
        else:
            connect.insert(collection, default_entity, partition_name=tag_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_type(self, connect, collection, get_field_type):
        """
        target: test insert with invalid field
        method: insert with invalid field type
        expected: raise exception
        """
        field_type = get_field_type
        tmp_entity = ut.update_field_type(copy.deepcopy(default_entity), 'float', field_type)
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_value(self, connect, collection, get_field_int_value):
        """
        target: test insert with invalid field
        method: insert with invalid field value
        expected: raise exception
        """
        field_value = get_field_int_value
        tmp_entity = ut.update_field_type(copy.deepcopy(default_entity), 'int64', field_value)
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_entity_value(self, connect, collection, get_field_vectors_value):
        """
        target: test insert with invalid entity
        method: insert with invalid entity value
        expected: raise exception
        """
        tmp_entity = copy.deepcopy(default_entity)
        src_vector = tmp_entity[-1]["values"]
        src_vector[0][1] = get_field_vectors_value
        with pytest.raises(Exception):
            connect.insert(collection, tmp_entity)


class TestInsertInvalidBinary(object):
    """
    Test inserting vectors with invalid collection names
    """

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_collection_name(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_tag_name(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_field_name(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_field_type(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_field_int_value(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_ints()
    )
    def get_entity_id(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_vectors()
    )
    def get_field_vectors_value(self, request):
        yield request.param

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_name(self, connect, binary_collection, get_field_name):
        """
        target: test insert with invalid field name
        method: insert with invalid field name
        expected: raise exception
        """
        tmp_entity = ut.update_field_name(copy.deepcopy(default_binary_entity), "int64", get_field_name)
        with pytest.raises(Exception):
            connect.insert(binary_collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_value(self, connect, binary_collection, get_field_int_value):
        tmp_entity = ut.update_field_type(copy.deepcopy(default_binary_entity), 'int64', get_field_int_value)
        with pytest.raises(Exception):
            connect.insert(binary_collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_entity_value(self, connect, binary_collection, get_field_vectors_value):
        """
        target: test insert with invalid scenario
        method: insert with invalid field entity
        expected: raise exception
        """
        tmp_entity = copy.deepcopy(default_binary_entity)
        src_vectors = tmp_entity[-1]["values"]
        src_vectors[0] = get_field_vectors_value
        with pytest.raises(Exception):
            connect.insert(binary_collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_ids_invalid(self, connect, binary_id_collection, get_entity_id):
        """
        target: test insert, with using customize ids, which are not int64
        method: create collection and insert entities in it
        expected: raise exception
        """
        entity_id = get_entity_id
        ids = [entity_id for _ in range(default_nb)]
        with pytest.raises(Exception):
            connect.insert(binary_id_collection, default_binary_entities, ids)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_type(self, connect, binary_collection, get_field_type):
        """
        target: test insert with invalid field type
        method: insert with invalid field type
        expected: raise exception
        """
        field_type = get_field_type
        tmp_entity = ut.update_field_type(copy.deepcopy(default_binary_entity), 'int64', field_type)
        with pytest.raises(Exception):
            connect.insert(binary_collection, tmp_entity)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_field_entities_value(self, connect, binary_collection, get_field_vectors_value):
        """
        target: test insert with invalid field
        method: insert with invalid field value
        expected: raise exception
        """
        tmp_entities = copy.deepcopy(default_binary_entities)
        src_vector = tmp_entities[-1]["values"]
        src_vector[1] = get_field_vectors_value
        with pytest.raises(Exception):
            connect.insert(binary_collection, tmp_entities)

