from scipy.io import loadmat
import numpy as np


def load_data(data_dir):
    """
    load known/unknown data
    param data_dir: file dir
    :return: dict, {str: array} --> {variable_name: data}, useful data
    """	
    mat_data = loadmat(data_dir)
    if mat_data:
        buildin_keys = ['__version__', '__header__', '__globals__']
        buildin_data = {}
        for key in mat_data.copy():
            if key in buildin_keys:
                buildin_data[key] = mat_data[key]
                del mat_data[key]
            else:
                assert isinstance(mat_data[key], np.ndarray)
        return mat_data

    return None
	
	
def format_data(data_dict, key_x, key_y, num_labels=9):
    """
    format the data into data and label
    :param data_dict: dict, the data arrays
    :param key_x: str, name of data variable
    :param key_y: str, name of label variable
    :param num_labels: int, the number of labels/categories
    :return: two arrays, assume the axis 0 is the number of samples
    """
    assert isinstance(data_dict, dict)
    data_tmp = [None, None]
    for i, key in enumerate([key_x, key_y]):
        try:
            data_tmp[i] = data_dict[key]
        except KeyError:
            exit("Error: The variable '%s' does not exist" % key)

    x, y = data_tmp
    x_shape, y_shape = x.shape, y.shape
    assert x_shape[0] == y_shape[0]

    # make x float32
    x = x.astype(np.float32)
	
    # make y to be one-hot labels
    y = np.squeeze(y)
    y_one_hot = np.zeros((len(y), num_labels))
    y_one_hot[np.arange(len(y)), y - 1] = 1
	
    return x, y_one_hot
	

def seed_train_test(x, y, percentage=.3, random_seed=2020):
    if random_seed:
        np.random.seed(int(random_seed))
    num_samples = x.shape[0]
    assert num_samples == y.shape[0]

    # randomly sample testing data
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    idx_split = int(np.round(num_samples * percentage))
    indices_te = indices[:idx_split]
    indices_tr = indices[idx_split:]
    x_te, y_te = x[indices_te, ...], y[indices_te, ...]
    x_tr, y_tr = x[indices_tr, ...], y[indices_tr, ...]
    return x_tr, y_tr, x_te, y_te
