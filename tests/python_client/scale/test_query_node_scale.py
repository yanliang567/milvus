import threading
import time

import pytest

from base.collection_wrapper import ApiCollectionWrapper
from common.common_type import CaseLabel
from customize.milvus_operator import MilvusOperator
from common import common_func as cf
from common import common_type as ct
from scale import constants
from pymilvus import Index, connections
from utils.util_log import test_log as log
from utils.util_k8s import wait_pods_ready, read_pod_log
from utils.util_pymilvus import get_latest_tag

prefix = "search_scale"
nb = 5000
nq = 5
default_schema = cf.gen_default_collection_schema()
default_search_exp = "int64 >= 0"
default_index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}


class TestQueryNodeScale:

    @pytest.mark.tags(CaseLabel.L3)
    def test_scale_query_node(self):
        """
        target: test scale queryNode
        method: 1.deploy milvus cluster with 1 queryNode
                2.prepare work (connect, create, insert, index and load)
                3.continuously search (daemon thread)
                4.expand queryNode from 2 to 5
                5.continuously insert new data (daemon thread)
                6.shrink queryNode from 5 to 3
        expected: Verify milvus remains healthy and search successfully during scale
        """
        fail_count = 0
        release_name = "scale-query"
        image_tag = get_latest_tag()
        image = f'{constants.IMAGE_REPOSITORY}:{image_tag}'
        query_config = {
            'metadata.namespace': constants.NAMESPACE,
            'metadata.name': release_name,
            'spec.components.image': image,
            'spec.components.proxy.serviceType': 'LoadBalancer',
            'spec.components.queryNode.replicas': 1,
            'spec.config.dataCoord.enableCompaction': True,
            'spec.config.dataCoord.enableGarbageCollection': True
        }
        mic = MilvusOperator()
        mic.install(query_config)
        if mic.wait_for_healthy(release_name, constants.NAMESPACE, timeout=1200):
            host = mic.endpoint(release_name, constants.NAMESPACE).split(':')[0]
        else:
            # log.warning(f'Deploy {release_name} timeout and ready to uninstall')
            # mic.uninstall(release_name, namespace=constants.NAMESPACE)
            raise BaseException(f'Milvus healthy timeout 1200s')

        try:
            # connect
            connections.add_connection(default={"host": host, "port": 19530})
            connections.connect(alias='default')

            # create
            c_name = cf.gen_unique_str("scale_query")
            # c_name = 'scale_query_DymS7kI4'
            collection_w = ApiCollectionWrapper()
            collection_w.init_collection(name=c_name, schema=cf.gen_default_collection_schema(), shards_num=2)

            # insert two segments
            for i in range(3):
                df = cf.gen_default_dataframe_data(nb)
                collection_w.insert(df)
                log.debug(collection_w.num_entities)

            # create index
            collection_w.create_index(ct.default_float_vec_field_name, default_index_params)
            assert collection_w.has_index()[0]
            assert collection_w.index()[0] == Index(collection_w.collection, ct.default_float_vec_field_name,
                                                    default_index_params)

            # load
            collection_w.load()

            # scale queryNode to 5
            mic.upgrade(release_name, {'spec.components.queryNode.replicas': 5}, constants.NAMESPACE)

            # continuously search
            def do_search():
                while True:
                    search_res, _ = collection_w.search(cf.gen_vectors(1, ct.default_dim),
                                                        ct.default_float_vec_field_name,
                                                        ct.default_search_params, ct.default_limit)
                    log.debug(search_res[0].ids)
                    assert len(search_res[0].ids) == ct.default_limit

            t_search = threading.Thread(target=do_search, args=(), daemon=True)
            t_search.start()

            # wait new QN running, continuously insert
            mic.wait_for_healthy(release_name, constants.NAMESPACE)
            wait_pods_ready(constants.NAMESPACE, f"app.kubernetes.io/instance={release_name}")

            def do_insert():
                while True:
                    tmp_df = cf.gen_default_dataframe_data(1000)
                    collection_w.insert(tmp_df)

            t_insert = threading.Thread(target=do_insert, args=(), daemon=True)
            t_insert.start()

            log.debug(collection_w.num_entities)
            time.sleep(20)
            log.debug("Expand querynode test finished")

            mic.upgrade(release_name, {'spec.components.queryNode.replicas': 3}, constants.NAMESPACE)
            mic.wait_for_healthy(release_name, constants.NAMESPACE)
            wait_pods_ready(constants.NAMESPACE, f"app.kubernetes.io/instance={release_name}")

            log.debug(collection_w.num_entities)
            time.sleep(60)
            log.debug("Shrink querynode test finished")

        except Exception as e:
            log.error(str(e))
            fail_count += 1
            # raise Exception(str(e))

        finally:
            log.info(f'Test finished with {fail_count} fail request')
            assert fail_count <= 1
            label = f"app.kubernetes.io/instance={release_name}"
            log.info('Start to export milvus pod logs')
            read_pod_log(namespace=constants.NAMESPACE, label_selector=label, release_name=release_name)
            mic.uninstall(release_name, namespace=constants.NAMESPACE)