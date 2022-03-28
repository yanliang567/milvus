import time
import os
import random
from common.common_func import gen_vectors


def gen_row_based_json_file(row_file, rows, dim):
    with open(row_file, "w") as f:
        f.write("{")
        f.write("\n")
        f.write('"rows":[')
        f.write("\n")
        for i in range(rows):
            vector = gen_vectors(1, dim)
            f.write('{"id":"' + str(i) + '",')
            f.write('"vector":' + ",".join(str(x) for x in vector) + "},")
            f.write("\n")
        f.write("]")
        f.write("\n")
        f.write("}")
        f.write("\n")


def gen_column_base_json_file(col_file, rows, dim):
    with open(col_file, "w") as f:
        f.write("{")
        f.write("\n")
        f.write('"uid":[' + ",".join(str(i) for i in range(rows)) + "]")
        f.write("\n")
        vectors = gen_vectors(rows, dim)
        f.write('"vector":' + ",".join(str(x) for x in vectors))
        f.write("\n")
        f.write("}")
        f.write("\n")


if __name__ == '__main__':
    dim = 4
    rows = 2

    file = "/tmp/row_data.json"
    gen_row_based_json_file(row_file=file, rows=rows, dim=dim)

    # file = "/tmp/col_data.json"
    # gen_column_base_json_file(col_file=file, rows=rows, dim=dim )

