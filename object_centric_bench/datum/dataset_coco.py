from pathlib import Path
import json
import pickle as pkl
import time

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .utils import (
    rgb_segment_to_index_segment,
    index_segment_to_bbox,
    draw_segmentation_np,
)


class MSCOCO(ptud.Dataset):
    """
    Common Objects in COntext  https://cocodataset.org
    """

    def __init__(
        self,
        data_file,
        instance: bool,  # only things
        extra_keys=["bbox", "segment", "clazz"],
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
            # print(txn.get(b"__keys__"))
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        self.instance = instance
        self.extra_keys = extra_keys
        self.transform = transform

    def __getitem__(self, index, compact=True):
        """
        - image: shape=(c=3,h,w), uint8
        - bbox: shape=(n,c=4), float32
        - segment: shape=(h,w), uint8
        - clazz: shape=(n,), uint8
        - isthing: shape=(n,), bool
        """
        # load sample pack
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample1 = {}

        # load image and segment
        image0 = cv2.cvtColor(
            cv2.imdecode(  # cvtColor will unify images to 3 channels safely
                np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB,
        )
        image = pt.from_numpy(image0).permute(2, 0, 1)
        sample1["image"] = image

        def bytes2uint8(x):
            if isinstance(x, bytes):
                x = np.frombuffer(x, dtype=np.uint8)
            return x
        
        if "segment" in self.extra_keys:
            # print("keys:", sample0.keys())
            # print(sample0["segment"])
            segment = pt.from_numpy(
                cv2.imdecode(bytes2uint8(sample0["segment"]), cv2.IMREAD_GRAYSCALE)
            )
            sample1["segment"] = segment

            # load bbox and clazz for set prediction
            if "bbox" in self.extra_keys:
                bbox = pt.from_numpy(sample0["bbox"])
                sample1["bbox"] = bbox
            if "clazz" in self.extra_keys:
                clazz = pt.from_numpy(sample0["clazz"])
                sample1["clazz"] = clazz

        # conduct transformation
        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            # merge stuff into the background, idx=1
            # 0: bg or no-annotat; >0: fg (things & stuff for panoptic, things only for instance)
            isthing = pt.from_numpy(sample0["isthing"])
            segment = sample2["segment"]
            if self.instance:  # set all stuff indexes to 0 as the background
                stuff_idxs = (~isthing).nonzero() + 1
                segment[pt.isin(segment, stuff_idxs)] = 0
                sample2["segment"] = segment

            # remove invalid bbox and clazz
            isvalid = pt.zeros_like(isthing)
            sidxs0 = segment.unique().tolist()
            sidxs0 = list(set(sidxs0) - set(range(MSCOCO.OFFSET)))
            sidxs0.sort()
            isvalid[pt.from_numpy(np.array(sidxs0, "int64")) - MSCOCO.OFFSET] = 1
            if "bbox" in self.extra_keys:
                sample2["bbox"] = sample2["bbox"][isvalid]
            if "clazz" in self.extra_keys:
                sample2["clazz"] = sample2["clazz"][isvalid]

            # compact segment idxs to be continuous
            if compact:
                # seg2 = sample2["segment"]  # not work for segment with (sporadic) offset
                # sidxs, seg2 = seg2.unique(return_inverse=True)
                # seg2 = seg2.reshape(seg2.shape).byte()
                # delta = len(set(range(MSCOCO.OFFSET)) - set(sidxs.numpy().tolist()))
                # seg2 += delta
                segment = sample2["segment"]
                sidxs = segment.unique().tolist()
                cnt = MSCOCO.OFFSET
                for sidx in sidxs:
                    if sidx in range(MSCOCO.OFFSET):  # 0: bg or no-annotat
                        continue
                    segment[segment == sidx] = cnt
                    cnt += 1
                # assert (seg2 == segment).all()
                sample2["segment"] = segment

        return sample2

    def __len__(self):
        return len(self.idxs)

    OFFSET = 1  # 0 reserved for bg, i.e., no annotation for panoptic

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/mscoco"),
        dst_dir=Path("coco"),
    ):
        """
        Download dataset MSCOCO:
        - 2017 Train images [118K/18GB] http://images.cocodataset.org/zips/train2017.zip
        - 2017 Val images [5K/1GB] http://images.cocodataset.org/zips/val2017.zip

        Structure dataset as follows and run it!
        - annotations
          - panoptic_coco_categories.json  # download from https://github.com/cocodataset/panopticapi
          - panoptic_train2017.json
          - panoptic_train2017
            - *.png
          - panoptic_val2017.json
          - panoptic_val2017
            - *.png
        - tain2017
          - *.jpg
        - val2017
          - *.jpg
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        category_file = src_dir / "annotations" / "panoptic_coco_categories.json"
        with open(category_file, "r") as f:
            categories = json.load(f)
        categories = {category["id"]: category for category in categories}

        splits = dict(
            train=["train2017", "annotations/panoptic_train2017"],
            val=["val2017", "annotations/panoptic_val2017"],
        )

        for split, [image_dn, segment_dn] in splits.items():
            print(split, image_dn, segment_dn)

            annotation_file = src_dir / f"{segment_dn}.json"
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb.open(
                str(dst_file),
                map_size=1024**4,
                subdir=False,
                readonly=False,
                meminit=False,
            )

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            # https://github.com/cocodataset/panopticapi/blob/master/converters/panoptic2detection_coco_format.py
            for cnt, annotat in enumerate(annotations):
                sids0 = [_["id"] for _ in annotat["segments_info"]]
                assert len(sids0) == len(set(sids0)) < 256
                sinfo0 = dict(zip(sids0, annotat["segments_info"]))

                fn = annotat["file_name"].split(".")[0]
                image_file = src_dir / image_dn / f"{fn}.jpg"
                segment_file = src_dir / segment_dn / f"{fn}.png"

                with open(image_file, "rb") as f:
                    image_b = f.read()
                segment_bgr = cv2.imread(str(segment_file))  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment0 = (
                    (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2).astype("int32")
                )
                sidxs = np.unique(segment0).tolist()
                sidxs = list(set(sidxs) - set(range(MSCOCO.OFFSET)))
                sidxs.sort()
                assert set(sids0) == set(sidxs)

                segment = np.zeros_like(segment0, "uint8")  # (h,w)
                both_side = np.tile(segment.shape[:2][::-1], 2).astype("float32")
                bbox = []
                clazz = []
                isthing = []
                # 0: bg or no-annotat; >0: fg (things & stuff for panoptic, things only for instance)
                for si, sidx in enumerate(sidxs):
                    segment[segment0 == sidx] = si + MSCOCO.OFFSET
                    bb = sinfo0[sidx]["bbox"]  # xywh
                    bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]  # ltrb
                    ci = sinfo0[sidx]["category_id"]
                    it = categories[ci]["isthing"]
                    assert it in [0, 1]  # always true
                    bbox.append(bb)
                    clazz.append(ci)
                    isthing.append(it)
                bbox = np.array(bbox, "float32").reshape(-1, 4)  # in case no elements
                # bbox = index_segment_to_bbox(segment).reshape(-1, 4)
                bbox = bbox / both_side  # whwh
                clazz = np.array(clazz, "uint8")
                isthing = np.array(isthing, "bool")

                # image = cv2.imdecode(  # there are some grayscale images
                #     np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR
                # )
                # print(clazz)
                # print(isthing)
                # __class__.visualiz(image, bbox, segment, clazz, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert (
                    bbox.ndim == 2 and bbox.shape[1] == 4 and bbox.dtype == np.float32
                )
                assert segment.ndim == 2 and segment.dtype == np.uint8
                assert clazz.ndim == 1 and clazz.dtype == np.uint8
                assert isthing.ndim == 1 and isthing.dtype == np.bool
                assert (
                    len(set(np.unique(segment).tolist()) - set(range(MSCOCO.OFFSET)))
                    == bbox.shape[0]
                    == clazz.shape[0]
                    == isthing.shape[0]
                )

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    bbox=bbox,  # (n,c=4) float32 ltrb
                    # re-encoding consumes less space than segment_b
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                    clazz=clazz,  # (n,) uint8
                    isthing=isthing,  # (n,) bool
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

            print(f"total={cnt + 1}, time={time.time() - t0}")

    '''@staticmethod
    def convert_dataset(  # XXX coconut is exquisite but has too many segmentation masks to handle
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw"),
        dst_dir=Path("coco"),
    ):
        """
        Download datasets MSCOCO and COCONut (COCO with exquisite annotations):
        - 2017 Train images [118K/18GB] http://images.cocodataset.org/zips/train2017.zip
        - 2017 Val images [5K/1GB] http://images.cocodataset.org/zips/val2017.zip
        - Download dataset as zip (3 GB) https://www.kaggle.com/datasets/xueqingdeng/coconut

        Structure datasets as follows and run it!
        - mscoco
          - annotations
            - panoptic_coco_categories.json  # download from https://github.com/cocodataset/panopticapi
            - panoptic_train2017.json
            - panoptic_train2017
              - *.png
            - panoptic_val2017.json
            - panoptic_val2017
              - *.png
          - tain2017
            - *.jpg
          - val2017
            - *.jpg
        - coconut
          - annotations
            - annotations
              - coconut_s_panoptic.json
              - relabeled_coco_val_panoptic.json
          - coconut_s  # corresponds to coco train2017
            - panoptic
              - *.png
          - relabeled_coco_val  # corresponds to coco val2017
            - panoptic
              - *.png
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        category_file = src_dir / "mscoco/annotations/panoptic_coco_categories.json"
        with open(category_file, "r") as f:
            categories = json.load(f)
        categories = {category["id"]: category for category in categories}

        splits = dict(
            train=[
                "mscoco/train2017",
                "coconut/annotations/annotations/coconut_s_panoptic.json",
                "coconut/coconut_s/panoptic",
            ],
            val=[
                "mscoco/val2017",
                "coconut/annotations/annotations/relabeled_coco_val_panoptic.json",
                "coconut/relabeled_coco_val/panoptic",
            ],
        )

        for split, [image_dn, annotat_fn, segment_dn] in splits.items():
            print(split, image_dn, annotat_fn, segment_dn)

            annotation_file = src_dir / annotat_fn
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb.open(
                str(dst_file),
                map_size=1024**4,
                subdir=False,
                readonly=False,
                meminit=False,
            )

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            # https://github.com/cocodataset/panopticapi/blob/master/converters/panoptic2detection_coco_format.py
            for cnt, annotat in enumerate(annotations):
                sids0 = [_["id"] for _ in annotat["segments_info"]]
                assert len(sids0) == len(set(sids0)) < 256
                sinfo0 = dict(zip(sids0, annotat["segments_info"]))

                fn = annotat["file_name"].split(".")[0]
                image_file = src_dir / image_dn / f"{fn}.jpg"
                segment_file = src_dir / segment_dn / f"{fn}.png"

                with open(image_file, "rb") as f:
                    image_b = f.read()
                segment_bgr = cv2.imread(str(segment_file))  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment0 = (
                    (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2).astype("int32")
                )
                sidxs = np.unique(segment0).tolist()
                sidxs = list(set(sidxs) - set(range(MSCOCO.OFFSET)))
                sidxs.sort()
                if not (set(sids0) == set(sidxs)):
                    print(annotat["file_name"])  # some samples violate this
                assert max(sidxs) < 256  # some samples violate this

                segment = np.zeros_like(segment0, "uint8")  # (h,w)
                both_side = np.tile(segment.shape[:2][::-1], 2).astype("float32")
                bbox = []
                clazz = []
                isthing = []
                # 0: bg or no-annotat; >0: fg (things & stuff for panoptic, things only for instance)
                for si, sidx in enumerate(sidxs):
                    segment[segment0 == sidx] = si + MSCOCO.OFFSET
                    # bb = sinfo0[sidx]["bbox"]  # xywh
                    # bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]  # ltrb
                    ci = sinfo0[sidx]["category_id"]
                    it = categories[ci]["isthing"]
                    assert it in [0, 1]  # always true
                    # bbox.append(bb)
                    clazz.append(ci)
                    isthing.append(it)
                # bbox = np.array(bbox, "float32").reshape(-1, 4)  # in case no elements
                bbox = index_segment_to_bbox(segment).reshape(-1, 4)
                bbox = bbox / both_side  # whwh
                clazz = np.array(clazz, "uint8")
                isthing = np.array(isthing, "bool")

                # image = cv2.imdecode(  # there are some grayscale images
                #     np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR
                # )
                # print(clazz)
                # print(isthing)
                # __class__.visualiz(image, bbox, segment, clazz, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert (
                    bbox.ndim == 2 and bbox.shape[1] == 4 and bbox.dtype == np.float32
                )
                assert segment.ndim == 2 and segment.dtype == np.uint8
                assert clazz.ndim == 1 and clazz.dtype == np.uint8
                assert isthing.ndim == 1 and isthing.dtype == np.bool
                assert (
                    len(set(np.unique(segment).tolist()) - set(range(MSCOCO.OFFSET)))
                    == bbox.shape[0]
                    == clazz.shape[0]
                    == isthing.shape[0]
                )

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    bbox=bbox,  # (n,c=4) float32 ltrb
                    # re-encoding consumes less space than segment_b
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
                    clazz=clazz,  # (n,) uint8
                    isthing=isthing,  # (n,) bool
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

            print(f"total={cnt + 1}, time={time.time() - t0}")'''

    @staticmethod
    def visualiz(image, bbox=None, segment=None, clazz=None, wait=0):
        print("[INFO] visualiz() called")
        print(f"[INFO] image shape: {image.shape}, dtype: {image.dtype}")

        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8, "[ERROR] image must be (H,W,3) and uint8"

        img_vis = image.copy()

        if bbox is not None:
            print(f"[INFO] bbox shape: {bbox.shape}, dtype: {bbox.dtype}")
            assert bbox.ndim == 2 and bbox.shape[1] == 4 and bbox.dtype == np.float32, "[ERROR] bbox format incorrect"
            if bbox.shape[0] > 0:
                print("[INFO] Drawing bounding boxes...")
                bbox_scaled = bbox * np.tile(img_vis.shape[:2][::-1], 2)
                for box in bbox_scaled.astype("int"):
                    img_vis = cv2.rectangle(
                        img_vis, tuple(box[:2]), tuple(box[2:]), (0, 0, 0), 2
                    )

        print("[INFO] Showing image window...")
        print("img_vis: ", img_vis.shape)
        # cv2.imshow("i", img_vis)
        # cv2.imwrite("debug_image.png", img_vis)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        segment_viz = None
        if segment is not None:
            print(f"[INFO] segment shape: {segment.shape}, dtype: {segment.dtype}")
            assert segment.ndim == 2 and segment.dtype == np.uint8, "[ERROR] segment must be 2D uint8"
            print("[INFO] Drawing segmentation...")
            segment_viz = draw_segmentation_np(img_vis, segment, alpha=0.75)

            if clazz is not None:
                print(f"[INFO] clazz shape: {clazz.shape}, dtype: {clazz.dtype}")
                assert clazz.ndim == 1 and clazz.dtype == np.uint8, "[ERROR] clazz must be 1D uint8"
                if bbox is not None and bbox.shape[0] > 0:
                    nseg = np.unique(segment).tolist()
                    print(f"[INFO] segment unique ids: {nseg}")
                    nseg = list(set(nseg) - set(range(MSCOCO.OFFSET)))
                    print(f"[INFO] valid segment ids after offset removal: {nseg}")
                    nseg.sort()
                    assert len(nseg) == len(clazz), "[ERROR] Number of segments != number of classes"
                    for iseg, iclz in zip(nseg, clazz):
                        y, x = np.where(segment == iseg)
                        l = np.min(x)
                        t = np.min(y)
                        r = np.max(x)
                        b = np.max(y)
                        segment_viz = cv2.putText(
                            segment_viz,
                            f"{iclz}",
                            [int((l + r) / 2), int((t + b) / 2)],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255] * 3,
                        )

            print("[INFO] Showing segmentation window...")
            # cv2.imshow("s", segment_viz)
            # cv2.imwrite("segment_viz.png", segment_viz)

        print("[INFO] Waiting for key...")
        cv2.waitKey(wait)
        print("[INFO] Done.")
        return img_vis, segment_viz

    # def visualiz(image, bbox=None, segment=None, clazz=None, wait=0):
    #     """
    #     - image: bgr format, shape=(h,w,c=3), uint8
    #     - bbox: both normalized ltrb, shape=(n,c=4), float32
    #     - segment: index format, shape=(h,w), uint8
    #     - clazz: shape=(n,), uint8
    #     """
    #     assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

    #     if bbox is not None and bbox.shape[0]:
    #         assert bbox.ndim == 2 and bbox.shape[1] == 4 and bbox.dtype == np.float32
    #         bbox = bbox * np.tile(image.shape[:2][::-1], 2)
    #         for box in bbox.astype("int"):
    #             image = cv2.rectangle(
    #                 image, tuple(box[:2]), tuple(box[2:]), (0, 0, 0), 2
    #             )

    #     cv2.imshow("i", image)

    #     segment_viz = None
    #     if segment is not None:
    #         assert segment.ndim == 2 and segment.dtype == np.uint8
    #         segment_viz = draw_segmentation_np(image, segment, alpha=0.75)

    #         if clazz is not None and bbox.shape[0]:
    #             assert clazz.ndim == 1 and clazz.dtype == np.uint8
    #             nseg = np.unique(segment).tolist()
    #             nseg = list(set(nseg) - set(range(MSCOCO.OFFSET)))
    #             nseg.sort()
    #             assert len(nseg) == len(clazz)
    #             for iseg, iclz in zip(nseg, clazz):
    #                 y, x = np.where(segment == iseg)
    #                 l = np.min(x)
    #                 t = np.min(y)
    #                 r = np.max(x)
    #                 b = np.max(y)
    #                 segment_viz = cv2.putText(
    #                     segment_viz,
    #                     f"{iclz}",
    #                     [int((l + r) / 2), int((t + b) / 2)],
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.5,
    #                     [255] * 3,
    #                 )

    #         cv2.imshow("s", segment_viz)

    #     cv2.waitKey(wait)
    #     return image, segment_viz
