# Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import random
import io
import av
import cv2
import sys
import decord
import imageio
from decord import VideoReader
import torch
import numpy as np
import math
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from eval.utils import *
from streambridge.mm_utils import resize_video_qwen

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


decord.bridge.set_bridge("torch")

def get_uniform_indices(num_frames, vlen, start=0, end=None):
    if end is None:
        end = vlen
    frame_indices = torch.linspace(start, end - 1, num_frames).long().tolist()
    return frame_indices

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None, stride=-1,
    ):
    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)
    else:
        start_index = 0
    
    if sample == 'uniform':
        frame_indices = get_uniform_indices(num_frames, vlen, start=start_index, end=vlen)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )
    
    try:
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    except decord.DECORDError as e:
        print(f'decord.DECORDError: {e}')
    except Exception as e:
        print(f'Exception: {e}')
    
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    
    return frames, frame_indices, float(fps), vlen, duration



import av

def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base

def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)

def read_frames_av(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None,
    ):
    reader = av.open(video_path)
    frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = torch.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, fps, vlen, duration

def get_internvl_video_stream(video_path, sampled_fps, fix_frame_num=None, max_frame_num=None):
    from transformers import CLIPImageProcessor
    from streambridge.mm_utils import process_anyres_video_genli
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image_processor.image_mean = [0.485, 0.456, 0.406]
    image_processor.image_std = [0.229, 0.224, 0.225]
    image_processor.do_resize = True
    image_processor.do_center_crop = True
    image_processor.size['shortest_edge'] = 448
    image_processor.crop_size = {"height": 448, "width": 448}

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    duration = len(vr) / float(vr.get_avg_fps())
    frame_num = int(duration*sampled_fps) if fix_frame_num == None else fix_frame_num
    if max_frame_num != None:
        frame_num = min(max_frame_num, frame_num)

    frame_indices = np.linspace(0, len(vr) - 1, frame_num, dtype=int)
    video = vr.get_batch(frame_indices).numpy()  
    images_all = [Image.fromarray(frame) for frame in video]
    frame_pixel_values = []
    for frame in images_all:
        frame = process_anyres_video_genli(frame, image_processor, fix_res=448)
        frame_pixel_values.append(frame)
    frame_pixel_values = torch.stack(frame_pixel_values, dim=0)
    frame_pixel_values = frame_pixel_values.squeeze(1) # [num_frames, 3, h, w]
    return frame_pixel_values, None

def get_qwen2_vl_video_stream(video_path, sampled_fps, fix_frame_num=None, max_frame_num=None):
    from transformers import Qwen2VLImageProcessor
    processor_path = os.environ.get(
        "QWEN2VL_PROCESSOR",
        os.environ.get("CKPT", "Qwen/Qwen2-VL-7B-Instruct"),
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(processor_path)
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    duration = len(vr) / float(vr.get_avg_fps())
    frame_num = int(duration*sampled_fps) if fix_frame_num == None else fix_frame_num
    if max_frame_num != None:
        frame_num = min(max_frame_num, frame_num)

    frame_indices = np.linspace(0, len(vr) - 1, frame_num, dtype=int)
    video = vr.get_batch(frame_indices).numpy()  
    images_all = [Image.fromarray(frame) for frame in video]
    frame_pixel_values = []
    frame_grid_thw = []
    for frame in images_all:
        frame = resize_video_qwen(frame)
        image_outputs = image_processor(frame)
        frame_pixel_values.append(torch.tensor(image_outputs['pixel_values']))
        frame_grid_thw.append(image_outputs['image_grid_thw'][0])
    frame_pixel_values = torch.stack(frame_pixel_values, dim=0) # [num_frames, hw, dim]
    frame_grid_thw = torch.tensor(np.array(frame_grid_thw)) # [num_frames, 3]
    return frame_pixel_values, frame_grid_thw

def get_oryx_video_stream(video_path, sampled_fps, fix_frame_num=None, max_frame_num=None):
    from transformers import CLIPImageProcessor
    from streambridge.mm_utils import process_anyres_video_genli
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image_processor.image_mean = [0.5, 0.5, 0.5]
    image_processor.image_std = [0.5, 0.5, 0.5]
    image_processor.do_resize = False
    image_processor.do_center_crop = False

    vr = VideoReader(video_path, num_threads=1)
    duration = len(vr) / float(vr.get_avg_fps())
    frame_num = int(duration*sampled_fps) if fix_frame_num == None else fix_frame_num
    if max_frame_num != None:
        frame_num = min(max_frame_num, frame_num)

    frame_indices = np.linspace(0, len(vr) - 1, frame_num, dtype=int)
    video = vr.get_batch(frame_indices).numpy()  
    images_all = [Image.fromarray(frame) for frame in video]

    frame_pixel_values = []
    for frame in images_all:
        frame = process_anyres_video_genli(frame, image_processor)
        frame_pixel_values.append(frame)
    frame_pixel_values = torch.stack(frame_pixel_values, dim=0)
    frame_pixel_values = frame_pixel_values.squeeze(1) # [num_frames, 3, h, w]
    return frame_pixel_values

def get_llava_ov_video_stream(video_path, sampled_fps, fix_frame_num=None, max_frame_num=None):
    image_processor = frame_transform(image_size=384, mean=0.5, std=0.5)
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    frame_num = int(duration*sampled_fps) if fix_frame_num == None else fix_frame_num
    if max_frame_num != None:
        frame_num = min(max_frame_num, frame_num)

    pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
        video_path=video_path,
        num_frames=frame_num,
        sample='middle',
    )
    frame_pixel_values = torch.tensor(
        np.array([image_processor(pixel_values[i]) for i in range(pixel_values.shape[0])])
    )
    return frame_pixel_values
