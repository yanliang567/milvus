

def findkeys(node, kv):
    # refer to https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x


def update_key_value(node, modify_k, modify_v):
    # update the value of modify_k to modify_v
    if isinstance(node, list):
        for i in node:
            update_key_value(i, modify_k, modify_v)
    elif isinstance(node, dict):
        if modify_k in node:
            node[modify_k] = modify_v
        for j in node.values():
            update_key_value(j, modify_k, modify_v)
    return node


if __name__ == "__main__":
    d = { "id" : "abcde",
        "key1" : "blah",
        "key2" : "blah blah",
        "nestedlist" : [
        { "id" : "qwerty",
            "nestednestedlist" : [
            { "id" : "xyz", "keyA" : "blah blah blah" },
            { "id" : "fghi", "keyZ" : "blah blah blah" }],
            "anothernestednestedlist" : [
            { "id" : "asdf", "keyQ" : "blah blah" },
            { "id" : "yuiop", "keyW" : "blah" }] } ] }
    print(list(findkeys(d, 'id')))
    update_key_value(d, "none_id", "ccc")
    print(d)
