import threading
import time
from multiprocessing import Pool, Process
import pytest
from utils import util_pymilvus as ut
from common.constants import default_entities, default_fields
from common.common_type import CaseLabel
from utils.util_log import test_log as log

# TIMEOUT = 120
default_nb = ut.default_nb
default_tag = ut.default_tag


class TestCreateBase:
    """
    ******************************************************************
      The following cases are used to test `create_partition` function
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="skip temporarily for debug")
    # @pytest.mark.timeout(600)
    def test_create_partition_limit(self, connect, collection, args):
        """
        target: test create partitions, check status returned
        method: call function: create_partition for 4097 times
        expected: exception raised
        """
        threads_num = 8
        threads = []
        if args["handler"] == "HTTP":
            pytest.skip("skip in http mode")

        def create(connect, threads_num):
            for i in range(ut.max_partition_num // threads_num):
                tag_tmp = ut.gen_unique_str()
                connect.create_partition(collection, tag_tmp)

        for i in range(threads_num):
            m = ut.get_milvus(host=args["ip"], port=args["port"], handler=args["handler"])
            t = threading.Thread(target=create, args=(m, threads_num))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        tag_tmp = ut.gen_unique_str()
        with pytest.raises(Exception) as e:
            connect.create_partition(collection, tag_tmp)

    @pytest.mark.tags(CaseLabel.L0)
    def test_create_partition_repeat(self, connect, collection):
        """
        target: test create partition, check status returned
        method: call function: create_partition
        expected: status ok
        """
        connect.create_partition(collection, default_tag)
        try:
            connect.create_partition(collection, default_tag)
        except Exception as e:
            code = getattr(e, 'code', "The exception does not contain the field of code.")
            assert code == 1
            message = getattr(e, 'message', "The exception does not contain the field of message.")
            assert message == "CreatePartition failed: partition name = %s already exists" % default_tag
        assert ut.compare_list_elements(connect.list_partitions(collection), [default_tag, '_default'])

    @pytest.mark.tags(CaseLabel.L2)
    def test_create_partition_name_name_none(self, connect, collection):
        """
        target: test create partition, tag name set None, check status returned
        method: call function: create_partition
        expected: status ok
        """
        tag_name = None
        try:
            connect.create_partition(collection, tag_name)
        except Exception as e:
            assert e.args[0] == "`partition_name` value None is illegal"

class TestShowBase:

    """
    ******************************************************************
      The following cases are used to test `list_partitions` function
    ******************************************************************
    """
    @pytest.mark.tags(CaseLabel.L0)
    def test_list_partitions(self, connect, collection):
        """
        target: test show partitions, check status and partitions returned
        method: create partition first, then call function: list_partitions
        expected: status ok, partition correct
        """
        connect.create_partition(collection, default_tag)
        assert ut.compare_list_elements(connect.list_partitions(collection), [default_tag, '_default'])

    @pytest.mark.tags(CaseLabel.L0)
    def test_list_partitions_no_partition(self, connect, collection):
        """
        target: test show partitions with collection name, check status and partitions returned
        method: call function: list_partitions
        expected: status ok, partitions correct
        """
        res = connect.list_partitions(collection)
        assert ut.compare_list_elements(res, ['_default'])

    @pytest.mark.tags(CaseLabel.L0)
    def test_show_multi_partitions(self, connect, collection):
        """
        target: test show partitions, check status and partitions returned
        method: create partitions first, then call function: list_partitions
        expected: status ok, partitions correct
        """
        tag_new = ut.gen_unique_str()
        connect.create_partition(collection, default_tag)
        connect.create_partition(collection, tag_new)
        res = connect.list_partitions(collection)
        assert ut.compare_list_elements(res, [default_tag, tag_new, '_default'])


class TestHasBase:

    """
    ******************************************************************
      The following cases are used to test `has_partition` function
    ******************************************************************
    """
    @pytest.fixture(
        scope="function",
        params=ut.gen_invalid_strs()
    )
    def get_tag_name(self, request):
        yield request.param

    @pytest.mark.tags(CaseLabel.L0)
    def test_has_partition_a(self, connect, collection):
        """
        target: test has_partition, check status and result
        method: create partition first, then call function: has_partition
        expected: status ok, result true
        """
        connect.create_partition(collection, default_tag)
        res = connect.has_partition(collection, default_tag)
        log.info(res)
        assert res

    @pytest.mark.tags(CaseLabel.L0)
    def test_has_partition_multi_partitions(self, connect, collection):
        """
        target: test has_partition, check status and result
        method: create partition first, then call function: has_partition
        expected: status ok, result true
        """
        for tag_name in [default_tag, "tag_new", "tag_new_new"]:
            connect.create_partition(collection, tag_name)
        for tag_name in [default_tag, "tag_new", "tag_new_new"]:
            res = connect.has_partition(collection, tag_name)
            assert res

    @pytest.mark.tags(CaseLabel.L2)
    def test_has_partition_name_not_existed(self, connect, collection):
        """
        target: test has_partition, check status and result
        method: then call function: has_partition, with tag not existed
        expected: status ok, result empty
        """
        res = connect.has_partition(collection, default_tag)
        log.info(res)
        assert not res

    @pytest.mark.tags(CaseLabel.L2)
    def test_has_partition_collection_not_existed(self, connect, collection):
        """
        target: test has_partition, check status and result
        method: then call function: has_partition, with collection not existed
        expected: status not ok
        """
        collection_name = "not_existed_collection"
        try:
            connect.has_partition(collection_name, default_tag)
        except Exception as e:
            code = getattr(e, 'code', "The exception does not contain the field of code.")
            assert code == 1
            message = getattr(e, 'message', "The exception does not contain the field of message.")
            assert message == "HasPartition failed: can't find collection: %s" % collection_name

    @pytest.mark.tags(CaseLabel.L2)
    def test_has_partition_with_invalid_tag_name(self, connect, collection, get_tag_name):
        """
        target: test has partition, with invalid tag name, check status returned
        method: call function: has_partition
        expected: status ok
        """
        tag_name = get_tag_name
        connect.create_partition(collection, default_tag)
        with pytest.raises(Exception) as e:
            connect.has_partition(collection, tag_name)


class TestDropBase:

    """
    ******************************************************************
      The following cases are used to test `drop_partition` function
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    def test_drop_partition_repeatedly(self, connect, collection):
        """
        target: test drop partition twice, check status and partition if existed
        method: create partitions first, then call function: drop_partition
        expected: status not ok, no partitions in db
        """
        connect.create_partition(collection, default_tag)
        connect.drop_partition(collection, default_tag)
        time.sleep(2)
        try:
            connect.drop_partition(collection, default_tag)
        except Exception as e:
            code = getattr(e, 'code', "The exception does not contain the field of code.")
            assert code == 1
            message = getattr(e, 'message', "The exception does not contain the field of message.")
            assert message == "DropPartition failed: partition %s does not exist" % default_tag
        tag_list = connect.list_partitions(collection)
        assert default_tag not in tag_list

    @pytest.mark.tags(CaseLabel.L0)
    def test_drop_partition_create(self, connect, collection):
        """
        target: test drop partition, and create again, check status
        method: create partitions first, then call function: drop_partition, create_partition
        expected: status not ok, partition in db
        """
        connect.create_partition(collection, default_tag)
        assert ut.compare_list_elements(connect.list_partitions(collection), [default_tag, '_default'])
        connect.drop_partition(collection, default_tag)
        assert ut.compare_list_elements(connect.list_partitions(collection), ['_default'])
        time.sleep(2)
        connect.create_partition(collection, default_tag)
        assert ut.compare_list_elements(connect.list_partitions(collection), [default_tag, '_default'])


class TestNameInvalid(object):
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
    def get_collection_name(self, request):
        yield request.param

    @pytest.mark.tags(CaseLabel.L2)
    def test_drop_partition_with_invalid_collection_name(self, connect, collection, get_collection_name):
        """
        target: test drop partition, with invalid collection name, check status returned
        method: call function: drop_partition
        expected: status not ok
        """
        collection_name = get_collection_name
        connect.create_partition(collection, default_tag)
        with pytest.raises(Exception) as e:
            connect.drop_partition(collection_name, default_tag)

    @pytest.mark.tags(CaseLabel.L2)
    def test_drop_partition_with_invalid_tag_name(self, connect, collection, get_tag_name):
        """
        target: test drop partition, with invalid tag name, check status returned
        method: call function: drop_partition
        expected: status not ok
        """
        tag_name = get_tag_name
        connect.create_partition(collection, default_tag)
        with pytest.raises(Exception) as e:
            connect.drop_partition(collection, tag_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_list_partitions_with_invalid_collection_name(self, connect, collection, get_collection_name):
        """
        target: test show partitions, with invalid collection name, check status returned
        method: call function: list_partitions
        expected: status not ok
        """
        collection_name = get_collection_name
        connect.create_partition(collection, default_tag)
        with pytest.raises(Exception) as e:
            connect.list_partitions(collection_name)

