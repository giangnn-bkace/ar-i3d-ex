import numpy as np

def get_label_map(file):
    label_map = []
    with open(file) as f:
        for line in f.readlines():
            label_map.append(line.strip())
    return label_map


def trans_label(label_in, label_map):
    #print(type(label_in))
    if isinstance(label_in, str):
        return label_map.index(label_in)
    elif isinstance(label_in, np.int64):
        return label_map[label_in]
    return
