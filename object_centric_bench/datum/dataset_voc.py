from pathlib import Path
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


class PascalVOC(ptud.Dataset):
    """Visual Object Classes Challenge 2012 (VOC2012, train) + 2007 (val)
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit
    - http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    """

class PascalVOC(ptud.Dataset):
    """
    数据集类：用于加载 Pascal VOC 图像分割数据集（支持2012和2007版本）。
    数据存储为 LMDB 格式，通过 LMDB 高效读取每个样本。
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment"],
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
    ):
        # 构造函数：初始化数据集
        # data_file: LMDB 文件名
        # extra_keys: 额外需要读取的数据键（如分割掩码"segment"）
        # transform: 数据增强或变换函数，输入为样本字典
        # max_spare: LMDB 允许的最大空闲事务数
        # base_dir: 数据根目录（可选）

        if base_dir:
            data_file = base_dir / data_file  # 拼接根目录

        # 打开 LMDB 数据库，只读模式（高效读取，不加锁）
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )

        # 获取所有样本索引列表（LMDB的键列表，通常为图片文件名或序号）
        with self.env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))  # 反序列化索引列表

        self.extra_keys = extra_keys      # 需要额外读取的数据键
        self.transform = transform        # 样本变换函数

    def __getitem__(self, index, compact=True):
        """
        获取一个样本
        Args:
            index: 样本索引
            compact: 是否对分割标签做离散化（连续编号）
        Returns:
            sample2: 经过变换的样本字典
        样本结构:
        - image: (3, h, w), uint8 格式图像
        - segment: (h, w), uint8 分割掩码
        """

        # 1. 从 LMDB 读取原始数据（sample0 是序列化后的字典）
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))

        sample1 = {}

        # 2. 解析图片，BGR转RGB，(h, w, c) -> torch (c, h, w)
        image0 = cv2.cvtColor(
            cv2.imdecode(
                np.frombuffer(sample0["image"], "uint8"), cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB,
        )
        image = pt.from_numpy(image0).permute(2, 0, 1)
        sample1["image"] = image

        # 3. 如需要，读取并解码分割掩码（灰度图），转为 torch 张量
        if "segment" in self.extra_keys:
            segment = pt.from_numpy(
                cv2.imdecode(sample0["segment"], cv2.IMREAD_GRAYSCALE)
            )
            sample1["segment"] = segment

        # 4. 对样本进行变换（如数据增强、归一化等）
        sample2 = self.transform(**sample1)

        # 5. 对分割标签做连续化（类别 ID 紧凑排序），便于后续训练
        if "segment" in self.extra_keys:
            if compact:
                segment = sample2["segment"]
                # unique返回所有值和每个像素的映射，用return_inverse=True实现连续编号
                segment = (
                    segment.unique(return_inverse=True)[1].reshape(segment.shape).byte()
                )
                sample2["segment"] = segment

        return sample2  # 返回处理后的样本字典


    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets_raw/pascalvoc/VOCdevkit"),
        dst_dir=Path("voc"),
    ):
        """
        Structure dataset as follows and run it!
        - VOC2012  # as training set
          - JPEGImages
            - *.jpg
          - SegmentationObject
            - *.png
        - VOC2007  # as validation set
          - JPEGImages
            - *.jpg
          - SegmentationObject
            - *.png
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train=[
                "VOC2012/JPEGImages",
                "VOC2012/SegmentationObject",
            ],
            val=[
                "VOC2007/JPEGImages",
                "VOC2007/SegmentationObject",
            ],
        )

        for split, [image_dn, segment_dn] in splits.items():
            image_path = src_dir / image_dn
            segment_path = src_dir / segment_dn
            segment_files = list(segment_path.iterdir())
            segment_files.sort()

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

            for cnt, segment_file in enumerate(segment_files):
                fn, ext = segment_file.name.split(".")
                assert ext == "png"
                image_file = image_path / f"{fn}.jpg"

                with open(image_file, "rb") as f:
                    image_b = f.read()
                segment_bgr = cv2.imread(str(segment_file))  # (h,w,c=3)
                segment_rgb = cv2.cvtColor(segment_bgr, cv2.COLOR_BGR2RGB)
                segment_rgb = np.where(  # ignore borderlines
                    segment_rgb == np.array([[[224, 224, 192]]]), 0, segment_rgb
                )
                segment = rgb_segment_to_index_segment(segment_rgb)

                # image = cv2.imdecode(np.frombuffer(image_b, "uint8"), cv2.IMREAD_COLOR)
                # __class__.visualiz(image, segment, wait=0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                assert type(image_b) == bytes
                assert segment.ndim == 2 and segment.dtype == np.uint8

                sample_dict = dict(
                    image=image_b,  # (h,w,c=3) bytes
                    segment=cv2.imencode(".webp", segment)[1],  # (h,w) uint8
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

    @staticmethod
    def visualiz(image, segment=None, wait=0):
        """
        - image: bgr format, shape=(h,w,c=3), uint8
        - segment: index format, shape=(h,w), uint8
        """
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        cv2.imshow("i", image)

        if segment is not None:
            assert segment.ndim == 2 and segment.dtype == np.uint8
            segment_viz = draw_segmentation_np(image, segment, alpha=0.75)
            cv2.imshow("s", segment_viz)

        cv2.waitKey(wait)
        return image, segment_viz
