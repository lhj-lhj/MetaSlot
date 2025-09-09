from pathlib import Path
import time

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .dataset import compress, decompress
from .utils import (
    even_resize_and_center_crop,
    rgb_segment_to_index_segment,
    index_segment_to_bbox,
    normaliz_for_visualiz,
    draw_segmentation_np,
)


class BiCoS(ptud.Dataset):
    """BiCoS: A Bi-level Co-Segmentation Method for Image Classification
    https://www.robots.ox.ac.uk/~vgg/data/bicos

    - Caltech/UCSD Birds
    https://www.robots.ox.ac.uk/~vgg/data/bicos/data/birds.tar
    - Weizmann Horses
    https://www.robots.ox.ac.uk/~vgg/data/bicos/data/horses.tar
    - Caltech 101: Airplanes
    https://www.robots.ox.ac.uk/~vgg/data/bicos/data/airplanes.tar
    - Caltech 101: Motorbikes
    https://www.robots.ox.ac.uk/~vgg/data/bicos/data/motorbikes.tar
    """

    RATIO = 4  # 4:1
    FLAGS = dict(
        train=lambda _: _ % BiCoS.RATIO != 0, val=lambda _: _ % BiCoS.RATIO == 0
    )

    def __init__(
        self,
        data_file,
        split,
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
            self_keys = decompress(txn.get(b"__keys__"))
        self.keys = [_ for i, _ in enumerate(self_keys) if __class__.FLAGS[split]]
        self.transform = transform

    def __getitem__(self, index):
        """
        - image: in shape (c=3,h,w), float32
        - segment: in shape (h,w), uint8
        """
        with self.env.begin(write=False) as txn:
            sample = decompress(txn.get(self.keys[index]))
        sample = dict(
            image=pt.from_numpy(sample["image"]).permute(2, 0, 1),
            segment=pt.from_numpy(sample["segment"]),
        )
        sample2 = self.transform(**sample)
        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def convert_dataset(
        src_dir=Path("/media/GeneralZ/Storage/Static/datasets/bicos/airplanes"),
        dst_dir=Path("airplanes"),
        resolut=(128, 128),  # spatial downsample: (?,?)->(h,w)
    ):
        """
        Structure dataset as follows and run it!
        - airplanes | birds | horses | motorbikes
          - gt
            - *.png
          - jpg
            - *.jpg
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        info_file = dst_dir / f"{resolut[0]}_{resolut[1]}"
        info_file.touch()
        assert resolut[0] == resolut[1]
        side = resolut[0]

        t0 = time.time()

        image_path = src_dir / "jpg"
        segment_path = src_dir / "gt"

        image_files = list(image_path.glob("*.jpg"))
        segment_files = list(segment_path.glob("*.png"))
        assert len(image_files) == len(segment_files)
        image_files.sort()
        segment_files.sort()

        dst_file = dst_dir / f"data.lmdb"
        lmdb_env = lmdb.open(
            str(dst_file),
            map_size=1024**4,
            subdir=False,
            readonly=False,
            meminit=False,
        )
        keys = []
        txn = lmdb_env.begin(write=True)

        cnt = 0
        for image_file, segment_file in zip(image_files, segment_files):
            assert image_file.name[:-3] == segment_file.name[:-3]

            image = cv2.imread(str(image_file)).astype("uint8")
            image = even_resize_and_center_crop(image, side)

            segment = cv2.imread(str(segment_file))
            if "horses" in src_dir.name:  # XXX
                segment = (segment > 200).astype("uint8")
            segment = even_resize_and_center_crop(
                segment, side, cv2.INTER_NEAREST_EXACT
            )
            segment, bbox = color_segment_to_index_segment_and_bbox(segment)
            idxs = set(np.unique(segment))
            if len(idxs) != 2:
                print(image_file, idxs)
                continue

            # __class__.visualiz(image, segment, 0)

            sample_key = f"{cnt:06d}".encode("ascii")
            keys.append(sample_key)

            sample_dict = dict(
                image=image,  # (h,w,c=3)
                segment=segment,  # (h,w)
            )
            txn.put(sample_key, compress(sample_dict))

            if (cnt + 1) % 64 == 0:  # write_freq
                print(f"{(cnt+1):06d}")
                txn.commit()
                txn = lmdb_env.begin(write=True)

            cnt += 1

        txn.commit()
        txn = lmdb_env.begin(write=True)
        txn.put(b"__keys__", compress(keys))
        txn.commit()
        lmdb_env.close()

        print(f"total={cnt + 1}, time={time.time() - t0}")

    @staticmethod
    def visualiz(image, segment=None, wait=0):
        if isinstance(image, pt.Tensor):
            image = image.permute(1, 2, 0).cpu().contiguous().numpy()
        image = normaliz_for_visualiz(image)
        image = (image * 127.5 + 127.5).astype("uint8")
        cv2.imshow("i", image)

        if segment is not None:
            if isinstance(segment, pt.Tensor):
                segment = segment.cpu().numpy()
            segment = draw_segmentation_np(image, segment, 0.6)
            cv2.imshow("s", segment)

        cv2.waitKey(wait)
        return image, segment
