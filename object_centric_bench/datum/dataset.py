import pickle as pkl

import lmdb
import torch.utils.data as ptud
import tqdm
import zstd as zs


DataLoader = ptud.DataLoader


ChainDataset = ptud.ChainDataset


ConcatDataset = ptud.ConcatDataset


# StackDataset = ptud.StackDataset


def compress(_):
    """https://gregoryszorc.com/blog/2017/03/07/better-compression-with-zstandard"""
    return zs.compress(pkl.dumps(_, pkl.HIGHEST_PROTOCOL))


def decompress(_):
    return pkl.loads(zs.decompress(_))


def merge_lmdbs(lmdb_files: dict, save_dir):
    """lmdb_files=dict(train=lmdb_file_train,val=lmdb_file_val)"""
    dst_file = save_dir / ("_".join([str(_) for _ in lmdb_files.keys()]) + ".lmdb")
    lmdb_env = lmdb.open(
        str(dst_file), map_size=1024**4, subdir=False, readonly=False, meminit=False
    )

    keys2 = []
    txn2 = lmdb_env.begin(write=True)
    cnt = 0

    for key, file in lmdb_files.items():
        sub_lmdb_env = lmdb.open(
            str(file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=4,
            lock=False,
        )

        with sub_lmdb_env.begin(write=False) as txn:
            sub_keys = decompress(txn.get(b"__keys__"))

        for sub_key in sub_keys:
            with sub_lmdb_env.begin(write=False) as txn:
                sample = txn.get(sub_key)

            sub_key2 = f"{sub_key.decode('ascii')}_{key}".encode("ascii")
            keys2.append(sub_key2)
            txn2.put(sub_key2, sample)

            if (cnt + 1) % 64 == 0:  # write_freq
                print(f"{(cnt+1):06d}", sub_key2)
                txn2.commit()
                txn2 = lmdb_env.begin(write=True)
            cnt += 1

    txn2.commit()
    txn2 = lmdb_env.begin(write=True)
    txn2.put(b"__keys__", compress(keys2))
    txn2.commit()
    lmdb_env.close()


def count_total_mean_std(dataset, keys, dims):
    assert len(keys) == len(dims)
    statistic = {_: {"mean": 0, "var": 0} for _ in keys}
    for sample in tqdm.tqdm(dataset):
        for key, dim in zip(keys, dims):
            statistic[key]["mean"] += sample[key].float().mean(dim=dim).numpy()
            statistic[key]["var"] += sample[key].float().var(dim=dim).numpy()
    n = len(dataset)
    statistic = {
        k: {"mean": v["mean"] / n, "std": (v["var"] / n) ** 0.5}
        for k, v in statistic.items()
    }
    return statistic
