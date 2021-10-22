# import datetime
# from pymilvus import connections, utility
# from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
#
# # connect to milvus
# host = '10.98.0.7'
# port = 19530
# connections.add_connection(default={"host": host, "port": port})
# t0 = datetime.datetime.now()
# print(f"start connect: {t0}")
# conn = connections.connect(alias='default')
# t1 = datetime.datetime.now()
# print(f"end connect: {t1}, duration: {t1 - t0}")
# conn.list_collections()
#
# # create a collection with auto_id as primary field
# dim = 128
# auto_id = FieldSchema(name="auto_id", dtype=DataType.INT64,  description="auto primary id")
# age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
# embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
# schema = CollectionSchema(fields=[auto_id, age_field, embedding_field],
#                           auto_id=True, primary_field=auto_id.name,
#                           description="desc of collection")
# collection_name2 = "tutorial_2"
# collection2 = Collection(name=collection_name2, schema=schema)
#
# import random
# # insert data with auto primary id
# nb = 300
# ages = [random.randint(20, 40) for i in range(nb)]
# embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
# entities2 = [ages, embeddings]
# ins_res2 = collection2.insert(entities2)
#
# import pandas as pd
# df = pd.DataFrame({
#         "cus_id": [i for i in range(nb)],
#         "age": [random.randint(20, 40) for i in range(nb)],
#         "embedding": [[random.random() for _ in range(dim)] for _ in range(nb)]
#     })
# collection3, ins_res3 = Collection.construct_from_dataframe(
#                                 'tutorial3',
#                                 df,
#                                 primary_field='cus_id',
#                                 auto_id=False
#                                 )
#
# # query on auto primary field
# collection2.load()
# expr = f"auto_id in {ins_res2.primary_keys[0:5]}"       # only suport query on primary field by 'in'
# query_res2 = collection2.query(expr)
