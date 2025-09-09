from collections import defaultdict
from pathlib import Path
import json
import pickle as pkl
import time

from einops import rearrange, repeat
from pycocotools import mask as maskUtils
import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .utils import (
    normaliz_for_visualiz,
    draw_segmentation_np,
    calc_foreground_center_bbox,
    index_segment_to_bbox,
)


class YTVIS(ptud.Dataset):
    """(High-Quality) Youtube Video Instance Segmentation.

    Number of time steps distribution
    - train: {5: 2, 6: 1, 8: 2, 9: 6, 10: 11, 11: 3, 12: 8, 13: 10, 14: 10, 15: 10, 16: 8, 17: 8, 18: 33, 19: 235, 20: 343, 21: 1, 24: 2, 25: 3, 26: 3, 27: 1, 28: 11, 29: 94, 30: 206, 31: 2, 32: 5, 33: 2, 34: 3, 35: 23, 36: 632}
    - val: {8: 1, 10: 5, 11: 1, 14: 2, 15: 1, 16: 1, 17: 1, 18: 3, 19: 50, 20: 53, 24: 1, 25: 1, 27: 1, 28: 1, 29: 16, 30: 39, 34: 1, 35: 7, 36: 95}
    - test: {10: 3, 11: 1, 13: 1, 14: 4, 17: 2, 18: 5, 19: 37, 20: 52, 21: 1, 22: 1, 29: 21, 30: 39, 34: 1, 35: 4, 36: 108}

    Number of objects distribution
    - train: {1: 1143, 2: 902, 3: 184, 4: 68, 5: 14, 6: 8}
    - val: {1: 207, 2: 125, 3: 28, 4: 15, 5: 2}
    - test: {1: 202, 2: 146, 3: 41, 4: 13}
    """

    def __init__(
        self,
        data_file,
        extra_keys=["bbox", "segment"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
        ts=20,  # least number of time steps
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
            self_keys = pkl.loads(txn.get(b"__keys__"))

        self.keys = []
        print(f"[{__class__.__name__}] slicing samples in dataset...")
        t0 = time.time()
        for key in self_keys:
            with self.env.begin(write=False) as txn:
                sample = pkl.loads(txn.get(key))
            t = len(sample["video"])
            if t < ts:
                continue
            num = int(np.ceil(t / ts))  # split ts>20 into multiple
            for i in range(num):
                start = (i * ts) if (i + 1 < num) else (t - ts)
                end = start + ts
                if end > num:
                    start = end - ts
                self.keys.append([key, start])
        print(f"[{__class__.__name__}] {time.time() - t0}")

        self.extra_keys = extra_keys
        self.transform = transform
        self.ts = ts

    def __getitem__(self, index, compact=True):
        """
        video: in shape (t=20,c=3,h,w), uint8
        bbox: in shape (t,n,c=4), float32, ltrb, both side normalized, only foreground
        segment: in shape (t,h,w), uint8
        """
        key, start = self.keys[index]
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(key))
        sample1 = {}

        video0 = sample0["video"][start : start + self.ts]
        video = np.array(
            [
                cv2.imdecode(np.frombuffer(_, "uint8"), cv2.IMREAD_UNCHANGED)
                for _ in video0
            ]
        )
        video = pt.from_numpy(video).permute(0, 3, 1, 2)
        sample1["video"] = video

        if "bbox" in self.extra_keys:
            bbox0 = sample0["bbox"][start : start + self.ts]
            bbox = pt.from_numpy(bbox0)
            sample1["bbox"] = bbox

        if "segment" in self.extra_keys:
            segment0 = sample0["segment"][start : start + self.ts]
            segment = np.array(
                [cv2.imdecode(_, cv2.IMREAD_GRAYSCALE) for _ in segment0]
            )
            segment = pt.from_numpy(segment)
            sample1["segment"] = segment

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            if compact:
                segment = sample2["segment"]
                segment = (
                    segment.unique(return_inverse=True)[1].reshape(segment.shape).byte()
                )
                sample2["segment"] = segment

        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/ytvis0"),
        dst_dir=Path("ytvis"),
    ):
        """
        Convert the original folded images into LMDB files.

        The code is adapted from
        https://github.com/SysCV/vmt/blob/main/cocoapi_hq/PythonAPI/pycocotools/ytvos.py

        Download the original dataset from
        https://youtube-vos.org/dataset/vis -- Data Download -> 2019 version new
        -> https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get-data -- Image frames -> Google Drive
        -> https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz -- all_frames
        - train_all_frames.zip
        - val_all_frames.zip
        - test_all_frames.zip

        Download the high-quality annotation files from
        https://www.vis.xyz/data/hqvis -- Dataset Download -> Download Link
        -> https://drive.google.com/drive/folders/1ZU8_qO8HnJ_-vvxIAn8-_kJ4xtOdkefh

        Unzip these three zip files and move all video folders into one JPEGImages folder;
        also put the annotation files here
        - JPEGImages  # videos of train/val/test are all here
            - 0a2f2bd294
            - 0a7a2514aa
            ...
        - ytvis_hq-test.json
        - ytvis_hq-train.json
        - ytvis_hq-val.json

        Finally execute this function.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train="ytvis_hq-train.json",
            val="ytvis_hq-val.json",
            test="ytvis_hq-test.json",
        )
        video_fold = src_dir / "JPEGImages"

        for split, annot_fn in splits.items():
            print(split, annot_fn)
            annot_file = src_dir / annot_fn
            with open(annot_file, "r") as fi:
                annot = json.load(fi)

            video_infos = {}
            for vinfo in annot["videos"]:
                video_infos[vinfo["id"]] = vinfo

            track_infos = defaultdict(list)
            for tinfo in annot["annotations"]:
                track_infos[tinfo["video_id"]].append(tinfo)

            lmdb_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb.open(
                str(lmdb_file),
                map_size=1024**4,
                subdir=False,
                readonly=False,
                meminit=False,
            )

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            cnt = 0
            for vid, track_info in track_infos.items():
                if len(track_info) == 0:
                    continue

                frame_fns = video_infos[vid]["file_names"]  # (t,h,w,c)
                video = [(video_fold / _).read_bytes() for _ in frame_fns]

                t = len(track_info[0]["segmentations"])
                h = track_info[0]["height"]
                w = track_info[0]["width"]
                both_side = np.tile([w, h], 2).astype("float32")

                assert all(h == _["height"] for _ in track_info)
                assert all(w == _["width"] for _ in track_info)
                assert all(t == len(_["segmentations"]) for _ in track_info)
                assert t == len(video)

                segment = np.zeros([t, h, w], "uint8")
                bbox = np.zeros([t, len(track_info), 4], "float32")
                for j, track in enumerate(track_info):
                    assert j + 1 < 256
                    mask = __class__.rle_to_mask(track, h, w)
                    assert set(np.unique(mask)) <= {0, 1}
                    box = np.array(
                        [[0] * 4 if _ is None else _ for _ in track["bboxes"]],
                        "float32",
                    )
                    box[:, 2] += box[:, 0]  # xywh -> ltrb
                    box[:, 3] += box[:, 1]
                    segment = np.where(mask.astype("bool"), (j + 1), segment)
                    bbox[:, j, :] = box

                assert np.unique(segment).max() > 0
                assert bbox.max() > 0
                bbox = bbox / both_side  # ltbr, float32, both-size normalized

                # video = np.array(
                #     [
                #         cv2.imdecode(np.frombuffer(_, "uint8"), cv2.IMREAD_COLOR)
                #         for _ in video
                #     ]
                # )
                # __class__.visualiz(video, bbox, segment, 0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                sample_dict = dict(
                    video=video,  # (t,h,w,c=3), bytes
                    bbox=bbox,  # (t,n,c=4), ltbr, float32
                    segment=[  # (t,h,w), bytes
                        cv2.imencode(".webp", _)[1] for _ in segment
                    ],
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                cnt += 1

            txn.commit()
            print((time.time() - t0) / cnt)

            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

    @staticmethod
    def rle_to_mask(track, h, w):
        masks = []
        for frameId in range(len(track["segmentations"])):
            brle = track["segmentations"][frameId]
            if brle is None:  # not visible
                mask = np.zeros([h, w], "uint8")
            else:
                if type(brle) == list:  # polygon; merge parts belonging to one object
                    rles = maskUtils.frPyObjects(brle, h, w)
                    rle = maskUtils.merge(rles)
                elif type(brle["counts"]) == list:  # uncompress RLE
                    rle = maskUtils.frPyObjects(brle, h, w)
                else:  # ???
                    rle = brle
                mask = maskUtils.decode(rle)
            masks.append(mask)
        return np.array(masks)  # (t,h,w)

    @staticmethod
    def visualiz(video, bbox=None, segment=None, wait=0):
        """
        - video: bgr format, shape=(t,h,w,c=3), uint8
        - bbox: both side normalized ltrb, shape=(t,n,c=4), float32
        - segment: index format, shape=(t,h,w), uint8
        """
        assert video.ndim == 4 and video.shape[3] == 3 and video.dtype == np.uint8

        if bbox is not None and bbox.shape[0]:
            assert bbox.ndim == 3 and bbox.shape[2] == 4 and bbox.dtype == np.float32
            t, h, w, c = video.shape
            bbox[:, :, 0::2] *= w
            bbox[:, :, 1::2] *= h
            bbox = bbox.astype("int")

        if segment is not None:
            assert segment.ndim == 3 and segment.dtype == np.uint8

        c1 = (64, 127, 255)
        imgs = []
        segs = []

        for t, img in enumerate(video):
            if bbox is not None and len(bbox) > 0:
                for b in bbox[t]:
                    cv2.rectangle(img, b[:2], b[2:], color=c1)

            cv2.imshow("v", img)
            imgs.append(img)

            if segment is not None:
                seg = draw_segmentation_np(img, segment[t], alpha=0.75)
                cv2.imshow("s", seg)
                segs.append(seg)

            cv2.waitKey(wait)

        return imgs, segs
