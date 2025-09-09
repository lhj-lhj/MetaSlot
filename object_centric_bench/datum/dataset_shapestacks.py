from pathlib import Path
import time

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .dataset import compress, decompress
from .utils import (
    rgb_segment_to_index_segment,
    index_segment_to_bbox,
    even_resize_and_center_crop,
    normaliz_for_visualiz,
    draw_segmentation_np,
)


class ShapeStacks(ptud.Dataset):
    """ShapeStacks Dataset
    https://ogroth.github.io/shapestacks

    Number of objects distribution:
    - train: {2: 3020, 3: 4144, 4: 3276, 5: 1969, 6: 1209}
    - val: {2: 646, 3: 888, 4: 702, 5: 423, 6: 257}
    - test: {2: 646, 3: 888, 4: 702, 5: 423, 6: 257}
    """

    def __init__(
        self,
        data_file,
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
            self.keys = decompress(txn.get(b"__keys__"))
        self.transform = transform

    def __getitem__(self, index):
        """
        - image: in shape (c=3,h,w), float32
        - bbox: in shape (n,c=4), float32
        - smask: in shape (n,), bool, masking slots
        - segment: in shape (h,w), uint8
        """
        with self.env.begin(write=False) as txn:
            sample = decompress(txn.get(self.keys[index]))
        n, c = sample["bbox"].shape
        sample = dict(
            image=pt.from_numpy(sample["image"]).permute(2, 0, 1),
            bbox=pt.from_numpy(sample["bbox"]),
            smask=pt.ones([n], dtype=pt.bool),
            segment=pt.from_numpy(sample["segment"]),
        )
        sample2 = self.transform(**sample)
        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets/ShapeStacks"),
        dst_dir=Path("shapestacks"),
        resolut=(128, 128),  # spatial downsample
        frame_idx=0,  # only use one frame
    ):
        """
        Structure dataset as follows and run it!
        - shapestacks-iseg/shapestacks/recordings
          - env_blocks-easy-h=2-vcom=0-vpsf=0-v=1
            - *.map
          ...
        - shapestacks-meta/shapestacks/splits
          - default
            - eval.txt
            - test.txt
            - train.txt
        - shapestacks-rgb/shapestacks/recordings
          - env_blocks-easy-h=2-vcom=0-vpsf=0-v=1
            - *.png
          ...
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        info_file = dst_dir / f"{resolut[0]}_{resolut[1]}-{frame_idx}"
        info_file.touch()
        assert resolut[0] == resolut[1]
        side = resolut[0]

        split_path = src_dir / "shapestacks-meta/shapestacks/splits/default"
        video_path = src_dir / "shapestacks-rgb/shapestacks/recordings"
        segment_path = src_dir / "shapestacks-iseg/shapestacks/recordings"

        splits = dict(train="train", val="eval", test="test")
        for split, alias in splits.items():
            with open(split_path / f"{alias}.txt", "r") as f:
                lines = f.readlines()
            video_fns = [_.strip() for _ in lines]

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

            for cnt, video_fn in enumerate(video_fns):
                image_files = list((video_path / video_fn).glob("*.png"))
                image_files.sort()
                image_file = image_files[frame_idx]

                segment_files = list((segment_path / video_fn).glob("*.map"))
                segment_files.sort()
                segment_file = segment_files[frame_idx]

                assert "-".join(image_file.name[:-4].split("-")[-3:]) == "-".join(
                    segment_file.name[:-4].split("-")[-3:]
                )

                image = cv2.imread(str(image_file))
                assert image.shape[0] == image.shape[1]
                image = even_resize_and_center_crop(image, side)

                segment = cv2.imread(str(segment_file))
                segment = even_resize_and_center_crop(
                    segment, side, cv2.INTER_NEAREST_EXACT
                )
                segment, bbox = color_segment_to_index_segment_and_bbox(segment)

                # __class__.visualiz(image, bbox / side, segment, 0)

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                sample_dict = dict(
                    image=image,  # (h,w,c=3)
                    bbox=bbox / side,  # (n,c=4)
                    segment=segment,  # (h,w)
                )
                txn.put(sample_key, compress(sample_dict))

                if (cnt + 1) % 64 == 0:  # write_freq
                    print(f"{(cnt+1):06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", compress(keys))
            txn.commit()
            lmdb_env.close()

            print(f"total={cnt + 1}, time={time.time() - t0}")

    @staticmethod
    def visualiz(image, bbox=None, segment=None, wait=0):
        if isinstance(image, pt.Tensor):
            image = image.permute(1, 2, 0).cpu().contiguous().numpy()
        image = normaliz_for_visualiz(image)
        image = np.clip(image * 127.5 + 127.5, 0, 255).astype("uint8")

        if bbox is not None:
            if isinstance(bbox, pt.Tensor):
                bbox = bbox.cpu().numpy()
            bbox[:, 0::2] *= image.shape[1]
            bbox[:, 1::2] *= image.shape[0]
            for box in bbox.astype("int"):
                image = cv2.rectangle(image, box[:2], box[2:], (0, 0, 0))

        cv2.imshow("i", image)

        if segment is not None:
            if isinstance(segment, pt.Tensor):
                segment = segment.cpu().numpy()
            segment = draw_segmentation_np(image, segment, 0.6)
            cv2.imshow("s", segment)

        cv2.waitKey(wait)
        return image, segment
