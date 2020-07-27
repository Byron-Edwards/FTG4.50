import pickle
from collections import OrderedDict


def state_dict_trans(state_dict, file_path=None):
    new_dict = OrderedDict()
    for param_tensor in state_dict:
        new_dict[param_tensor] = state_dict[param_tensor].numpy()
    if file_path:
        f = open(file_path, "wb")
        pickle.dump(new_dict, f)
        f.close()
    return new_dict