from pathlib import Path
import io
import pickle as pkl
import time

import av
import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .dataset import decompress
from .utils import normaliz_for_visualiz, draw_segmentation_np


class Physion(ptud.Dataset):
    """Physion: Evaluating Physical Prediction from Vision in Humans and Machines

    https://github.com/cogtoolslab/physics-benchmarking-neurips2021
    """

    def __init__(
        self,
        data_file,
        keys=["video"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
            self.idxs_scene = pkl.loads(txn.get(b"__keys_scene__"))
        self.keys = keys
        self.transform = transform

    def __getitem__(self, index):
        """
        - video: in shape (t=24,c=3,h,w), uint8
        - clazz: in shape (), long
        """
        with self.env.begin(write=False) as txn:
            sample = pkl.loads(txn.get(self.idxs[index]))
        sample2 = {}
        if "video" in self.keys:
            vcap = av.open(io.BytesIO(sample["video"]))
            video = [_.to_ndarray(format="rgb24") for _ in vcap.decode(0)]
            video = np.array(video)  # (t,h,w,c)
            sample2["video"] = pt.from_numpy(video).permute(0, 3, 1, 2)
        if "clazz" in self.keys:
            sample2["clazz"] = pt.tensor(sample["clazz"]).long()
        sample3 = self.transform(**sample2)
        return sample3

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets/physion0"),
        dst_dir=Path("physion"),
        size=256,
    ):
        """
        Convert the original mp4 and csv files into one LMDB file to save I/O overhead.

        Download the following files.
        - rename to PhysionTest-Core.zip https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip
        - rename to PhysionTrain-Core.tar.gz https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/PhysionTrainMP4s.tar.gz
        - https://github.com/cogtoolslab/physics-benchmarking-neurips2021/blob/master/data/readout_labels.csv

        Then unzip and structure these files as below:
        - physion0
            - PhysionTest-Core
                - Collide
                - Contain
                ...
                - Support
                - labels.csv
            - PhysionTrain-Core
                - Collide_readout_MP4s
                - Collide_training_MP4s
                ...
                - Support_training_MP4s
                - readout_labels.csv

        Finally create a Python script with the following content at the project root, and execute it:
        ```python
        from object_centric_bench.datum import Physion
        Physion.convert_dataset()  # remember to change default paths to yours
        ```
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        info_file = dst_dir / f"size{size}"
        info_file.touch()
        split_dict = dict(
            train=dict(
                dynamics=["PhysionTrain-Core/[?]_training_MP4s/*.mp4", None],
                readout=[
                    "PhysionTrain-Core/[?]_readout_MP4s/*.mp4",
                    "PhysionTrain-Core/readout_labels.csv",
                ],
            ),
            val=dict(
                dynamics=["PhysionTest-Core/[?]/mp4s/*.mp4", None],
                readout=[
                    "PhysionTest-Core/[?]/mp4s-redyellow/*.mp4",
                    "PhysionTest-Core/labels.csv",
                ],
            ),
        )
        scenarios = [
            "Dominoes",
            "Support",
            "Collide",
            "Contain",
            "Drop",
            "Link",
            "Roll",
            "Drape",
        ]

        for split, task_dict in split_dict.items():
            for task, (src_path0, label_fn) in task_dict.items():
                if label_fn is None:
                    assert task == "dynamics"
                    label_dict = None
                else:
                    assert task == "readout"
                    label_file = src_dir / label_fn
                    label_str = label_file.read_text()
                    parts = [
                        _.strip().split(",") for _ in label_str.strip().split("\n")[1:]
                    ]
                    ks = [str(_[0]) for _ in parts]
                    vs = [bool(_[1]) for _ in parts]
                    ks2 = [_.replace("-redyellow", "") for _ in ks]
                    assert len(ks) == len(set(ks2))
                    label_dict = dict(zip(ks2, vs))

                dst_file = dst_dir / f"{split}-{task}.lmdb"
                lmdb_env = lmdb.open(
                    str(dst_file),
                    map_size=1024**3,
                    subdir=False,
                    readonly=False,
                    meminit=False,
                )
                keys = []
                keys_scene = []
                txn = lmdb_env.begin(write=True)
                t0 = time.time()

                for scene in scenarios:
                    print(split, task, scene)
                    src_path = src_path0.replace("[?]", scene)
                    src_files = list(src_dir.glob(src_path))
                    src_files.sort()
                    print(len(src_files))

                    for i, src_file in enumerate(src_files):
                        # vcap = cv2.VideoCapture(str(src_file))
                        # video = []
                        # while True:
                        #     flag, frame = vcap.read()
                        #     if not flag:
                        #         break
                        #     if frame.shape == (256, 256, 3):
                        #         pass
                        #     else:
                        #         assert frame.shape == (512, 512, 3)
                        #         frame = cv2.resize(frame, [256, 256])
                        #     video.append(frame)
                        # print(video[0].shape)
                        video = src_file.read_bytes()

                        if label_dict is None:
                            clazz = None
                        else:
                            lbl_key = src_file.name[: -8 if split == "train" else -4]
                            lbl_key = lbl_key.replace("-redyellow", "")
                            assert lbl_key in label_dict
                            clazz = label_dict[lbl_key]

                        # ``av.open.decode`` decodes 10x faster than ``cv2.imdecode`` from webp format
                        # if split == "val":
                        #     tv = time.time()
                        #     vcap = av.open(io.BytesIO(video))
                        #     frames = [
                        #         _.to_ndarray(format="rgb24") for _ in vcap.decode(0)
                        #     ]
                        #     # frames = [next(vcap.decode(0)).to_ndarray(format="rgb24")]
                        #     frames = np.stack(frames)  # (t,h,w,c)
                        #     print(time.time() - tv)
                        #     assert frames.shape[1:] in [(256, 256, 3), (512, 512, 3)]
                        #     if not frames.shape[1:] == (256, 256, 3):
                        #         print()
                        #     del frames, vcap
                        # __class__.visualiz(frames, wait=0)

                        sample_key = f"{i:06d}".encode("ascii")
                        keys.append(sample_key)
                        scene_key = scene.encode("ascii")
                        keys_scene.append(scene_key)

                        sample_dict = dict(
                            video=video,  # mp4 bytes (t=150,h=256,w=256,c=3)
                            # bbox=bbox,  # float32
                            # segment=segment,  # uint8
                            clazz=clazz,
                        )
                        # sample_dict = dict(
                        #     video=[cv2.imencode(".webp", _)[1] for _ in video],
                        #     clazz=clazz,
                        # )
                        txn.put(sample_key, pkl.dumps(sample_dict))

                        if (i + 1) % 64 == 0:  # write_freq
                            print(f"{i + 1:06d}")
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                txn.commit()
                txn = lmdb_env.begin(write=True)
                print(len(keys), len(keys_scene))
                txn.put(b"__keys__", pkl.dumps(keys))
                txn.put(b"__keys_scene__", pkl.dumps(keys_scene))
                txn.commit()
                lmdb_env.close()

                print((time.time() - t0) / (i + 1))  # 0.0298842241987586

    @staticmethod
    def visualiz(video, vnorm=None, wait=0):
        """No batch."""
        if isinstance(video, pt.Tensor):
            if vnorm is not None:  # broadcastable mean and std
                vmean, vstd = pt.tensor(vnorm[0]), pt.tensor(vnorm[1])
                video = video * vstd + vmean
                video = video.round().clip(0, 255).to(pt.uint8)
            video = video.permute(0, 2, 3, 1).cpu().contiguous().numpy()

        imgs = []

        for t, img in enumerate(video):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("v", img)
            imgs.append(img)

            cv2.waitKey(wait)

        return imgs


class PhysionSlotz(ptud.Dataset):

    def __init__(
        self,
        data_file,
        keys=["slotz"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
    ):
        super().__init__()
        if base_dir:
            data_file = base_dir / data_file
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        self.keys = keys
        self.transform = transform

    def __getitem__(self, index):
        """
        - slotz: in shape (t=48,n,c), float32
        - clazz: in shape (), long
        """
        with self.env.begin(write=False) as txn:
            sample = decompress(txn.get(self.idxs[index]))
        sample2 = {}
        if "slotz" in self.keys:
            sample2["slotz"] = pt.from_numpy(sample["slotz"]).float()
        if "clazz" in self.keys:
            sample2["clazz"] = pt.tensor(sample["clazz"]).long()
        sample3 = self.transform(**sample2)
        return sample3

    def __len__(self):
        return len(self.idxs)
