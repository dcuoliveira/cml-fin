import pickle

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(obj, path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj