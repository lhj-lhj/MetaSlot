from pathlib import Path
import time

import cv2
import imageio as iio
import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud

from .dataset import compress, decompress
from .utils import (
    normaliz_for_visualiz,
    draw_segmentation_np,
    even_resize_and_center_crop,
)


class Habitat(ptud.Dataset):
    """Habitat-Lab,
    https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md
    habitat-lab/0habitatlabquickstart-collectdata.py

    pointnav_mp3d
    ---
    cfg_file: benchmark/nav/pointnav/pointnav_mp3d.yaml
    Point goal navigation	MatterPort3D	pointnav_mp3d_v1.zip	data/datasets/pointnav/mp3d/v1/	datasets/pointnav/mp3d.yaml	400 MB

    objectnav_mp3d
    ---
    cfg_file: benchmark/nav/objectnav/objectnav_mp3d.yaml
    Object goal navigation	MatterPort3D	objectnav_mp3d_v1.zip	data/datasets/objectnav/mp3d/v1/	datasets/objectnav/mp3d.yaml	170 MB
    """

    def __init__(
        self,
        data_file,
        transform=lambda **_: _,
        max_spare=4,
        base_dir: Path = None,
        repeat=1,
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
        self.repeat = repeat

    def __getitem__(self, index):
        """
        video: in shape (t,c=3,h,w), float32
        depth: in shape (t,c=1,h,w), float32
        """
        index = index % len(self.keys)
        with self.env.begin(write=False) as txn:
            sample = decompress(txn.get(self.keys[index]))
        sample = dict(
            video=pt.from_numpy(sample["video"]).permute(0, 3, 1, 2),  # (t,c,h,w) uint8
            depth=pt.from_numpy(sample["depth"]),  # (t,h,w), float32
        )
        sample2 = self.transform(**sample)
        return sample2

    def __len__(self):
        return len(self.keys) * self.repeat

    def convert_dataset(
        # cfg_file="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        cfg_file="benchmark/nav/pointnav/pointnav_mp3d.yaml",
        # save_dir=Path("pointnav_habitat_test"),
        save_dir=Path("pointnav_mp3d"),
        resolut=[128, 128],
    ):
        """Generating samples manully. Random generation usually gets stuck.
        https://github.com/facebookresearch/habitat-lab
        """
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        import habitat

        print(HabitatSimActions._known_actions)
        # actions = {
        #     k: v
        #     for k, v in HabitatSimActions._known_actions.items()
        #     if "stop" not in k and "look" not in k
        # }
        actions = dict(
            w=HabitatSimActions.move_forward,
            a=HabitatSimActions.turn_left,
            d=HabitatSimActions.turn_right,
            # u=HabitatSimActions.look_up,
            # b=HabitatSimActions.look_down,
            # f=HabitatSimActions.stop
        )
        print(actions)

        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "data.lmdb"

        info_file = save_dir / f"{resolut[0]}_{resolut[1]}-filter_scene"
        info_file.touch()
        assert resolut[0] == resolut[1]
        side = resolut[0]

        config = habitat.get_config(cfg_file)
        rlenv = habitat.Env(config=config)
        print("Environment creation successful")

        dbenv = lmdb.open(
            str(save_file),
            map_size=1024**4,
            subdir=False,
            readonly=False,
            meminit=False,
        )
        keys = []
        txn = dbenv.begin(write=True)

        t0 = time.time()
        episodes = []
        for e in rlenv.episodes:
            if all(_.scene_id != e.scene_id for _ in episodes):
                episodes.append(e)
        print(len(rlenv.episodes))

        for episode in episodes:
            key = (
                f"{episode.episode_id}-{episode.scene_id.split('/')[-1].split('.')[0]}"
            )
            print(key)
            key = key.encode("ascii")
            keys.append(key)

            rlenv.current_episode = episode
            video, depth = __class__.run_episode_and_collect_data(rlenv, side, actions)
            print(video.shape, depth.shape)

            sample_dict = dict(
                video=video,  # (h,w,c=3), uint8
                depth=depth,  # (h,w), float32
            )
            txn.put(key, compress(sample_dict))
            txn.commit()
            txn = dbenv.begin(write=True)

        txn.commit()
        txn = dbenv.begin(write=True)
        txn.put(b"__keys__", compress(keys))
        txn.commit()
        dbenv.close()

        print(f"total={len(episodes)}, time={time.time() - t0}")  # 8000sec

    @staticmethod
    def run_episode_and_collect_data(rlenv, side, actions):
        video = []
        depth = []

        observations = rlenv.reset()
        print(observations.keys())
        cv2.imshow("RGB", __class__.transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("depth", observations["depth"])

        print("Agent stepping around inside environment.")

        count_steps = 0
        while not rlenv.episode_over:
            frame_rgb = even_resize_and_center_crop(observations["rgb"], side)
            video.append(frame_rgb)
            frame_d = even_resize_and_center_crop(observations["depth"], side)
            depth.append(frame_d[:, :, 0])

            action_key = chr(cv2.waitKey(0))
            # cv2.waitKey(1)
            # action_key = np.random.choice(list(actions.keys()))
            if action_key not in actions:
                print("invalid key")
                continue
            print(action_key)

            action = actions[action_key]
            observations = rlenv.step(action)
            count_steps += 1

            cv2.imshow("RGB", __class__.transform_rgb_bgr(observations["rgb"]))
            cv2.imshow("depth", observations["depth"])

        print("Episode finished after {} steps.".format(count_steps))

        video = np.stack(video)
        depth = np.stack(depth)
        return video, depth

    @staticmethod
    def transform_rgb_bgr(image):
        return image[:, :, [2, 1, 0]]

    @staticmethod
    def visualiz(
        video,
        depth,
        segment,
        wait=0,
        permute=True,
        uint8=True,
        save: Path = None,
        fps=10,
    ):
        if permute:
            video = video.permute(0, 2, 3, 1)
        if isinstance(video, pt.Tensor):
            video = video.cpu().numpy()
            depth = depth.cpu().numpy()
            segment = segment.cpu().numpy()
        if uint8:
            video = ((video.clip(-0.5, +0.5) + 0.5) * 255).astype("uint8")
        demo = []
        for frame_rgb, frame_d, frame_s in zip(video, depth, segment):
            cv2.imshow("rgb", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow("d", frame_d)
            # frame_rgb = (normaliz_for_visualiz(frame_rgb) * 255).astype("uint8")
            frame_s = draw_segmentation_np(frame_rgb, frame_s, 0.7)
            cv2.imshow("s", frame_s)
            cv2.waitKey(wait)

            frame = np.concatenate(
                [
                    (np.repeat(frame_d[:, :, None], 3, 2) * 255).astype("uint8"),
                    frame_rgb,
                    frame_s,
                ],
                1,
            )
            # print(frame.shape, frame.dtype)
            demo.append(frame)
        if save:
            iio.mimsave(save, demo, fps=fps)
