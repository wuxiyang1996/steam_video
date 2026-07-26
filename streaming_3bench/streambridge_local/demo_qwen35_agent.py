#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Run StreamBridge activation with a local Qwen3.5 video-clip responder."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.backends import cudnn


def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def load_activation_stream(video_path: str, start_s: float, stream_fps: float):
    from streambridge.utils import frame_transform

    transform = frame_transform(image_size=384, mean=0.5, std=0.5)
    try:
        import decord

        reader = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        source_fps = float(reader.get_avg_fps())
        duration = len(reader) / source_fps
        if start_s >= duration:
            raise ValueError(f"start_s={start_s} is outside video duration {duration:.2f}s")
        timestamps = np.arange(start_s, duration, 1.0 / stream_fps)
        frame_indices = np.clip(np.round(timestamps * source_fps).astype(int), 0, len(reader) - 1)
        frames = reader.get_batch(frame_indices).asnumpy()
    except ModuleNotFoundError:
        import cv2

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or stream_fps)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / source_fps
        if start_s >= duration:
            raise ValueError(f"start_s={start_s} is outside video duration {duration:.2f}s")
        timestamps = np.arange(start_s, duration, 1.0 / stream_fps)
        frame_indices = np.clip(np.round(timestamps * source_fps).astype(int), 0, total_frames - 1)
        decoded = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        frames = np.stack(decoded, axis=0)

    frame_tensor = torch.tensor(frames).permute(0, 3, 1, 2)
    activate_pixel_values = torch.stack([transform(frame_tensor[i]) for i in range(frame_tensor.shape[0])], dim=0)
    return activate_pixel_values, timestamps.tolist(), duration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default="assets/__an5-LdjFY.mp4")
    parser.add_argument("--qwen35-ckpt", default="/mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B")
    parser.add_argument("--activate-pretrained", default="/mnt/haibo/streambridge-opensource/llava-onevision-qwen2-0.5b-ov-hf-seperated")
    parser.add_argument(
        "--activate-ckpt",
        default="/mnt/haibo/streambridge-opensource/activation_0.5_ratio_anet_coin_yc2_s2s_fa_mhego_hacs_cha_et_llava-ov_epoch_5.pth",
    )
    parser.add_argument("--question", default="Recognize and highlight specific sequences of actions.")
    parser.add_argument("--activate-threshold", type=float, default=0.3)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--stream-fps", type=float, default=1.0)
    parser.add_argument("--clip-window-s", type=float, default=4.0)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames-per-clip", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--always-remind", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from eval.streaming_models.online_qwen35_vl import Qwen35_VL
    from streambridge.model.activate_videollm import Activate_VideoLLM

    init_seeds(42)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    activate_model = Activate_VideoLLM(
        dtype=dtype,
        model_path=args.activate_pretrained,
        attn_implementation="eager" if device.type == "cpu" else "flash_attention_2",
        lora=True,
        pooling_factor=4,
        threshold=args.activate_threshold,
    )
    activate_model.load_state_dict(torch.load(args.activate_ckpt, map_location="cpu"), strict=False)
    activate_model.to(device)
    activate_model.eval()

    streaming_model = Qwen35_VL(
        ckpt=args.qwen35_ckpt,
        video_path=args.video_path,
        dtype=dtype,
        stream_fps=args.stream_fps,
        clip_window_s=args.clip_window_s,
        video_fps=args.video_fps,
        video_max_frames_per_clip=args.video_max_frames_per_clip,
    )

    activate_pixel_values, timestamps, duration = load_activation_stream(
        args.video_path,
        start_s=args.start_s,
        stream_fps=args.stream_fps,
    )
    activate_pixel_values = activate_pixel_values.to(device)
    print(f"Loaded {len(timestamps)} stream frames from {duration:.2f}s video", flush=True)

    activate_model.decide_response({"text_inputs": [f"<|im_start|>user {args.question}<|im_end|>"]})
    if not args.always_remind:
        streaming_model.receive_user_input(args.question)

    generate_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    activate_window = 128
    last_activate_window = 0
    for i, timestamp_s in enumerate(timestamps):
        probs, decision = activate_model.decide_response({"pixel_values": activate_pixel_values[i].unsqueeze(0)})
        streaming_model.receive_one_frame(timestamp_s=timestamp_s)

        if decision == 1:
            if args.always_remind:
                streaming_model.receive_user_input(args.question)
            answer = streaming_model.response(**generate_kwargs)[0]
            activate_model.decide_response({"text_inputs": [f"<|im_start|>assistant\n{answer}<|im_end|>"]})
            print(f"{timestamp_s:.2f} seconds: {answer}", flush=True)

        if i - last_activate_window == activate_window:
            activate_model.past_embeds = None
            activate_model.decide_response({"text_inputs": [f"<|im_start|>user {args.question}<|im_end|>"]})
            last_activate_window = i


if __name__ == "__main__":
    main()
