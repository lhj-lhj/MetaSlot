from argparse import ArgumentParser
from pathlib import Path
import torch as pt
import shutil

from object_centric_bench.datum import DataLoader
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config


@pt.no_grad()
def eval_epoch(pack):
    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for batch in pack.dataset_v:
        pack.batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(pack.batch)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn(**pack)

        pack.metric = pack.metric_fn_v(**pack)

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]


def main(args):
    cfg_file = Path(args.cfg_file)
    data_path = Path(args.data_dir)
    ckpt_file = Path(args.ckpt_file)
    cfg = Config.fromfile(cfg_file)

    # --- dataset ---
    cfg.dataset_v.base_dir = data_path
    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    # --- model ---
    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    if ckpt_file:
        print(f"Loading checkpoint (strict) from {ckpt_file}")
        state = pt.load(ckpt_file, map_location="cpu", weights_only=False)
        if "state_dict" in state:   # 兼容不同保存格式
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            print("⚠️ Strict load warnings:")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

    model = model.cuda().eval()

    # --- loss & metric ---
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    metric_fn_v = MetricWrap(detach=True, **build_from_config(cfg.metric_fn_v))

    # --- prepare save dir ---
    save_dir = Path(args.save_dir) / cfg_file.stem   # e.g., save/dinosaur_r-coco
    save_dir.mkdir(parents=True, exist_ok=True)

    for cb in cfg.callback_v:
        if cb.type == "AverageLog":
            cb.log_file = str(save_dir / "eval.txt")
    callback_v = build_from_config(cfg.callback_v)

    # --- pack ---
    pack = Config({})
    pack.dataset_v = dataload_v
    pack.model = model
    pack.loss_fn = loss_fn
    pack.metric_fn_v = metric_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0
    pack.step_count = 0


    # --- run eval ---
    with pt.inference_mode(True):
        eval_epoch(pack)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="save")
    return parser.parse_args()


if __name__ == "__main__":
    pt._dynamo.config.suppress_errors = True
    main(parse_args())