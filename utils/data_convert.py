import tqdm
import numpy as np

def _sample_t_end_indices(
    data: np.ndarray,
    idx,
    x_len: int,
    y_len: int,
    align: str = "last_x",
) -> np.ndarray:

    idx = np.asarray(idx, dtype=np.int64)
    t = len(idx) - 1
    ends: list[int] = []
    mode = (align or "last_x").strip().lower()
    for i in range(t, 0, -1):
        if i - x_len - y_len >= 0:
            if mode in ("first_y", "y_start"):
                k = i - y_len
            elif mode in ("last_y", "y_end"):
                k = i - 1
            else:
                k = i - y_len - 1
            ends.append(int(idx[k]))
    return np.asarray(ends, dtype=np.int32)

def z_score(data):

    return (data - np.mean(data)) / np.std(data)

def generate_dataset(data, idx, x_len: int = 12, y_len: int = 12):

    res = data[idx]
    node_size = data.shape[1]
    t = len(idx)-1
    idic = 0
    x_index, y_index = [], []

    for i in tqdm.tqdm(range(t, 0, -1)):
        if i-x_len-y_len>=0:
            x_index.extend(list(range(i-x_len-y_len, i-y_len)))
            y_index.extend(list(range(i-y_len, i)))

    x_index = np.asarray(x_index)
    y_index = np.asarray(y_index)
    x = res[x_index].reshape((-1, x_len, node_size))
    y = res[y_index].reshape((-1, y_len, node_size))

    return x, y

def generate_samples(
    days,
    savepath,
    data,
    graph,
    train_rate=0.6,
    val_rate=0.2,
    test_rate=0.2,
    val_test_mix=False,
    x_len: int = 12,
    y_len: int = 12,
    attach_t_end: bool = False,
    t_end_align: str = "last_x",
    air_train_fraction=None,
):

    edge_index = np.array(list(graph.edges)).T
    del graph

    if any(p in ("PEMS", "PEMS03") for p in savepath.replace("\\", "/").split("/")):
        data = data[0:days*288, :]

    t, n = data.shape[0], data.shape[1]

    frac = float(air_train_fraction) if air_train_fraction is not None else float(train_rate)
    train_idx = [i for i in range(int(t * frac))]
    val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
    test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]

    train_x, train_y = generate_dataset(data, train_idx, x_len=x_len, y_len=y_len)
    val_x, val_y = generate_dataset(data, val_idx, x_len=x_len, y_len=y_len)
    test_x, test_y = generate_dataset(data, test_idx, x_len=x_len, y_len=y_len)

    if val_test_mix:
        val_test_x = np.concatenate((val_x, test_x), 0)
        val_test_y = np.concatenate((val_y, test_y), 0)
        val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
        np.random.shuffle(val_test_idx)
        val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
        test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

    train_x = z_score(train_x)
    val_x = z_score(val_x)
    test_x = z_score(test_x)

    save_kw = dict(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        edge_index=edge_index,
    )
    if attach_t_end:
        save_kw["train_t_end"] = _sample_t_end_indices(data, train_idx, x_len, y_len, t_end_align)
        save_kw["val_t_end"] = _sample_t_end_indices(data, val_idx, x_len, y_len, t_end_align)
        save_kw["test_t_end"] = _sample_t_end_indices(data, test_idx, x_len, y_len, t_end_align)

    np.savez_compressed(savepath, **save_kw)
    return {k: v for k, v in save_kw.items()}
