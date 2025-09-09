from io import BytesIO
import colorsys

import av
import cv2
import numpy as np
import torch as pt
import torchvision.utils as ptvu
import torch.nn.functional as ptnf


def normaliz_for_visualiz(image: np.ndarray):
    return (image - image.min()) / (image.max() - image.min())


def even_resize_and_center_crop(image: np.ndarray, size: int, interp=cv2.INTER_LINEAR):
    h0, w0 = image.shape[:2]
    ratio = size / min(h0, w0)
    image2 = cv2.resize(image, dsize=None, fx=ratio, fy=ratio, interpolation=interp)
    h2, w2 = image2.shape[:2]
    t = (h2 - size) // 2
    l = (w2 - size) // 2
    b = t + size
    r = l + size
    output = image2[t:b, l:r]
    return output


def calc_foreground_center_bbox(segment_index, haxis=-2, waxis=-1):
    """
    segment_index: in shape (..,h,w)
    """
    foreground = segment_index > 0
    _, y, x = np.where(foreground)
    l = x.min()
    t = y.min()
    r = x.max()
    b = y.max()
    cx, cy = (l + r) / 2, (t + b) / 2
    h, w = segment_index.shape[haxis], segment_index.shape[waxis]
    side = min(h, w)
    lobe = side / 2
    if h == w:
        bbox = [0, 0, w, h]
    elif h < w:
        bbox = [cx - lobe, 0, cx + lobe, h]
    else:  # h > w
        bbox = [0, cy - lobe, w, cy + lobe]
    bbox = np.round(bbox).astype("int32")
    if bbox[0] < 0:
        bbox[0] = 0
        bbox[2] = side
    if bbox[1] < 0:
        bbox[1] = 0
        bbox[3] = side
    if bbox[2] > w:
        bbox[2] = w
        bbox[0] = w - side
    if bbox[3] > h:
        bbox[3] = h
        bbox[1] = h - side
    assert np.all(bbox[:2] >= 0) and (bbox[2] <= w) and (bbox[3] <= h)
    assert np.all(bbox[2:] - bbox[:2] == np.array([side] * 2))
    return bbox


def rgb_segment_to_index_segment(segment_rgb: np.ndarray):
    """
    segment_rgb: shape=(h,w,c=3). r-g-b not b-g-r
    segment_idx: shape=(h,w)
    """
    assert segment_rgb.ndim == 3 and segment_rgb.dtype == np.uint8
    assert segment_rgb.shape[2] == 3
    segment0 = (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2)
    segment_idx = (  # exactly same as the old implementation for-loop-assign
        np.unique(segment0, return_inverse=True)[1]
        .reshape(segment0.shape)
        .astype("uint8")
    )
    return segment_idx


def index_segment_to_bbox(segment_idx: np.ndarray):
    """
    segment_idx: shape=(h,w)
    bbox: shape=(n,c=4), ltrb
    """
    assert segment_idx.ndim == 2 and segment_idx.dtype == np.uint8
    idxs = np.unique(segment_idx).tolist()
    idxs.sort()
    if 0 in idxs:
        idxs.remove(0)  # not include the bbox for background
    bbox = np.zeros([len(idxs), 4], dtype="float32")
    for i, idx in enumerate(idxs):
        y, x = np.where(segment_idx == idx)
        bbox[i, 0] = np.min(x)  # left
        bbox[i, 1] = np.min(y)  # top
        bbox[i, 2] = np.max(x)  # right
        bbox[i, 3] = np.max(y)  # bottom
    return bbox


def generate_spectrum_colors(num_color):
    spectrum = []
    for i in range(num_color):
        hue = i / float(num_color)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        spectrum.append([int(255 * c) for c in rgb])
    return np.array(spectrum, dtype="uint8")  # (n,c=3)


def draw_segmentation_np(
    image: np.ndarray, segment: np.ndarray, max_num=0, alpha=0.5, colors=None
):
    """
    image: in shape (h,w,c)
    segment: in shape (h,w)
    """
    if not max_num:
        max_num = int(segment.max() + 1)
    if colors is None:
        colors = generate_spectrum_colors(max_num)  # len(np.unique(segment))
    mask = ptnf.one_hot(pt.from_numpy(segment.astype("int64")), max_num)
    image2 = ptvu.draw_segmentation_masks(
        image=pt.from_numpy(image).permute(2, 0, 1),
        masks=mask.bool().permute(2, 0, 1),
        alpha=alpha,
        colors=colors.tolist(),
    )
    return image2.permute(1, 2, 0).numpy()


class VideoCodec:
    """
    lossless encoding and decoding for videos

    - 3uint8: video (t,h,w,c=3) dtype=uint8 => no wrap, rgb24 => libx264rgb
    - 1uint16: depth (t,h,w,c=1) dtype=uint16 => wrap as rgb24 => libx264rgb

    Example
    ---
    ```
    video0 = np.random.randint(0, 255, [24, 256, 256, 3]).astype("uint8")
    # video0 = video0[:, :, :, :1].astype("uint16")  # 1uint16
    video0 = video0[:, :, :, :2].astype("uint16")  # 2uint16

    t20 = time()
    # buffer = VideoCodec.encode_3uint8(video0, fps)
    # buffer = VideoCodec.encode_1uint16(video0, fps)
    buffer = VideoCodec.encode_xuint16(video0, fps)
    # print(time() - t20, video0.nbytes, len(buffer.getvalue()))
    print(time() - t20, video0.nbytes, sum(len(_.getvalue()) for _ in buffer))

    t20 = time()
    # video2 = VideoCodec.decode_3uint8(buffer)
    # video2 = VideoCodec.decode_1uint16(buffer)
    video2 = VideoCodec.decode_xuint16(buffer)
    print(time() - t20)
    assert (video0 == video2).all()
    ```
    """

    @staticmethod
    def wrap_1uint16_as_rgb24(zero: np.ndarray) -> np.ndarray:
        t, h, w, c = zero.shape
        assert c == 1 and zero.dtype == np.uint16
        cr = (zero[:, :, :, 0] >> 8).astype("uint8")  # (t,h,w)
        cg = zero[:, :, :, 0].astype("uint8")
        cb = np.zeros_like(cg)
        wrap = np.stack([cr, cg, cb], axis=-1)  # (t,h,w,c=3)
        return wrap

    @staticmethod
    def wrap_rgb24_as_1uint16(zero: np.ndarray) -> np.ndarray:
        t, h, w, c = zero.shape
        assert c == 3 and zero.dtype == np.uint8
        c0 = zero[:, :, :, 0].astype("uint16") << 8 | zero[:, :, :, 1]  # (t,h,w)
        wrap = c0[:, :, :, None]  # (t,h,w,c=1)
        return wrap

    @staticmethod
    def encode_video_into_buffer(video: np.ndarray, pixfmt: str, fps: int) -> BytesIO:
        t, h, w, c = video.shape
        assert video.dtype == np.uint8
        if pixfmt == "rgb24":
            assert c == 3
            codec = "libx264rgb"
            stream_options = {
                "qp": "0",  # enforce encoder to do lossless encoding
                "tune": "fastdecode",
                "preset": "ultrafast",
            }
        elif pixfmt == "gray":
            assert c == 1
            codec = "ffv1"
            stream_options = {
                "level": "3",  # Faster decoding, lossless
                "coder": "1",  # Range coder for exact lossless encoding
            }
        else:
            raise NotImplementedError

        buffer = BytesIO()
        container = av.open(buffer, mode="w", format="avi")

        stream = container.add_stream(codec, rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = pixfmt
        stream.options = stream_options

        for frame_data in video:  # create a PyAV VideoFrame from the numpy array (RGB)
            frame = av.VideoFrame.from_ndarray(frame_data, format=pixfmt)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)

        container.close()
        buffer.seek(0)  # rewind to the start
        return buffer

    @staticmethod
    def decode_video_from_buffer(buffer: BytesIO) -> np.ndarray:
        container = av.open(buffer, mode="r", format="avi")

        frames = []
        # decode all frames from the first video stream
        for frame in container.decode(video=0):
            # convert frame to numpy array
            frame_array = frame.to_ndarray()  # format="rgb24"
            frames.append(frame_array)
        # list(map(...)) shows little speedup

        video = np.stack(frames, axis=0)  # (t,h,w,c)
        if video.ndim == 3:
            video = video[:, :, :, None]
        t, h, w, c = video.shape
        assert c in [1, 3] and video.dtype == np.uint8
        return video

    @staticmethod
    def encode_3uint8(video: np.ndarray, fps: int) -> BytesIO:
        return __class__.encode_video_into_buffer(video, "rgb24", fps)

    @staticmethod
    def decode_3uint8(buffer: BytesIO) -> np.ndarray:
        return __class__.decode_video_from_buffer(buffer)

    @staticmethod
    def encode_1uint8(video: np.ndarray, fps: int) -> BytesIO:
        return __class__.encode_video_into_buffer(video, "gray", fps)

    @staticmethod
    def decode_1uint8(buffer: BytesIO) -> np.ndarray:
        return __class__.decode_video_from_buffer(buffer)

    @staticmethod
    def encode_1uint16(zero: np.ndarray, fps: int) -> BytesIO:
        wrap = __class__.wrap_1uint16_as_rgb24(zero)  # assert inside
        buff = __class__.encode_3uint8(wrap, fps)
        return buff

    @staticmethod
    def decode_1uint16(buff: BytesIO) -> np.ndarray:
        wrap = __class__.decode_3uint8(buff)
        zero = __class__.wrap_rgb24_as_1uint16(wrap)  # assert inside
        return zero

    @staticmethod
    def encode_xuint16(zero: np.ndarray, fps: int) -> list:
        zero = np.split(zero, zero.shape[-1], axis=-1)
        buff = [__class__.encode_1uint16(_, fps) for _ in zero]
        return buff

    @staticmethod
    def decode_xuint16(buff: list) -> np.ndarray:
        zero = [__class__.decode_1uint16(_) for _ in buff]
        zero = np.concatenate(zero, axis=-1)  # (t,h,w,x)
        return zero
