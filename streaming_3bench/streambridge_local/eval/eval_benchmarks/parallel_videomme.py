#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torch.multiprocessing as mp
import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from eval.utils import *
from eval.streaming_models.online_qwen35_vl import Qwen35_VL


def result_path(filename):
    result_dir = os.environ.get("RESULT_DIR", "eval/results")
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, filename)


def video_duration_s(video_path):
    try:
        from decord import VideoReader

        reader = VideoReader(video_path)
        fps = float(reader.get_avg_fps() or 0)
        if fps > 0:
            return len(reader) / fps
    except Exception:
        pass
    try:
        import cv2

        capture = cv2.VideoCapture(video_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        if fps > 0:
            return frames / fps
    except Exception:
        pass
    return float(os.environ.get("VIDEOMME_OBSERVATION_END_S", 60.0))


def evaluate(rank, stream_data_split):
    torch.cuda.set_device(rank) 
    device = f'cuda:{rank}'
    sampled_fps = 1
    fix_frame_num = None

    if 'MAX_FRAME_NUM' in os.environ:
        max_frame_num = int(os.environ['MAX_FRAME_NUM'])
    else:
        max_frame_num = None
        
    if 'MAX_IMG_TOKEN' in os.environ:
        max_image_token = int(os.environ['MAX_IMG_TOKEN'])
    else:
        max_image_token = 32*1024

    if 'POOLING_FACTOR' in os.environ:
        pooling_factor = int(os.environ['POOLING_FACTOR'])
    else:
        pooling_factor = 4

    if 'CKPT' in os.environ:
        ckpt = os.environ['CKPT']
    else:
        ckpt = None

    model_name = os.environ['MODEL']

    oryx_process = False
    qwen_process = False
    qwen35_process = False
    llava_process = False
    intern_process = False

    if model_name == 'oryx':
        from eval.streaming_models.online_oryx import Oryx_1_5

        oryx_process = True
        model = Oryx_1_5(
            dtype=torch.bfloat16,
            img_size=384,
            pooling_factor=pooling_factor,
            ckpt = ckpt,
            max_image_token = max_image_token,
        )
    elif model_name == 'llava_ov':
        from eval.streaming_models.online_llava_ov import LLaVA_OV

        llava_process = True
        model = LLaVA_OV(
            dtype=torch.bfloat16,
            img_size=384,
            pooling_factor=pooling_factor,
            ckpt = ckpt,
            max_image_token = max_image_token,
        )
    elif model_name == 'qwen2vl':
        from eval.streaming_models.online_qwen2_vl import Qwen2_VL

        qwen_process = True
        model = Qwen2_VL(
            dtype=torch.bfloat16,
            img_size=384,
            pooling_factor=pooling_factor,
            ckpt = ckpt,
            max_image_token = max_image_token,
        )
    elif model_name == 'qwen35vl':
        qwen35_process = True
        model = Qwen35_VL(
            ckpt=ckpt,
            video_path="",
            dtype=torch.bfloat16,
            device_map={"": device},
            stream_fps=sampled_fps,
            clip_window_s=float(os.environ.get("CLIP_WINDOW_S", 4.0)),
            video_fps=float(os.environ.get("QWEN35_VIDEO_FPS", 2.0)),
            video_max_frames_per_clip=int(os.environ.get("QWEN35_VIDEO_MAX_FRAMES_PER_CLIP", 8)),
        )
    else:
        raise ValueError(f"Unsupported MODEL={model_name}")

    if not qwen35_process:
        model.to(device)
    res = []

    for item in tqdm(stream_data_split, desc=f"GPU {rank}"):
        video_path = item['video_file']
        question = item['question']
        type = item['type']
        answer = item['answer']

        if qwen35_process:
            model.video_path = video_path
            model.reset()
            frame_pixel_values = None
            frame_grid_thw = None
        else:
            from eval.video_utils import get_llava_ov_video_stream, get_oryx_video_stream, get_qwen2_vl_video_stream

            try:
                if oryx_process:
                    frame_pixel_values = get_oryx_video_stream(video_path, sampled_fps, fix_frame_num, max_frame_num).to(device)
                    frame_grid_thw = None
                elif qwen_process:
                    frame_pixel_values, frame_grid_thw = get_qwen2_vl_video_stream(video_path, sampled_fps, fix_frame_num, max_frame_num)
                    frame_pixel_values = frame_pixel_values.to(device)
                    frame_grid_thw = frame_grid_thw.to(device)
                elif llava_process:
                    frame_pixel_values = get_llava_ov_video_stream(video_path, sampled_fps, fix_frame_num, max_frame_num).to(device)
                    frame_grid_thw = None
            except Exception:
                print("Decord Error: ", video_path)
                continue

        generate_kwargs = {
            "do_sample": False,
            "num_beams": 1, 
            "min_length": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 128,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
                visible_until_s = float(os.environ.get("VIDEOMME_OBSERVATION_END_S", video_duration_s(video_path)))
                try:
                    if qwen35_process:
                        model.receive_one_frame(timestamp_s=visible_until_s)
                    else:
                        for j in range(frame_pixel_values.shape[0]):
                            if frame_grid_thw is None:
                                model.receive_one_frame(frame_pixel_values[j].unsqueeze(0))
                            else:
                                model.receive_one_frame(frame_pixel_values[j].unsqueeze(0), frame_grid_thw[j].unsqueeze(0))

                    model.receive_user_input(question)
                    output_text = model.response(**generate_kwargs)
                    pred_text = output_text[0]
                    error_text = None
                except Exception as exc:
                    pred_text = ""
                    error_text = f"{type(exc).__name__}: {exc}"

                res.append({
                    'video': video_path,
                    'type': type,
                    'question': question,
                    'pred': pred_text,
                    'answer': answer,
                    'model_name': model_name,
                    'ckpt': str(ckpt),
                    'ok': error_text is None,
                    'error': error_text,
                    'streaming_visible_until_s': visible_until_s if qwen35_process else None,
                    'streambridge_mode': 'qwen35vl_causal_clip' if qwen35_process else model_name,
                    'clip_window_s': float(os.environ.get("CLIP_WINDOW_S", 4.0)) if qwen35_process else None,
                    'qwen35_video_fps': float(os.environ.get("QWEN35_VIDEO_FPS", 2.0)) if qwen35_process else None,
                    'qwen35_video_max_frames_per_clip': int(os.environ.get("QWEN35_VIDEO_MAX_FRAMES_PER_CLIP", 8)) if qwen35_process else None,
                })

        if qwen35_process:
            model.reset()
        else:
            model.past_embeds = model.intialize_system_prompts()
            model.modality_indicators = [0 for i in range(model.past_embeds.shape[1])]
            model.cache = []

        save_json(res, result_path(f'videomme_eval_gpu{rank}.json'))

if __name__ == "__main__":

    anno_path = os.environ['ANNO_PATH']
    video_path = os.environ['VIDEO_PATH']
    data = load_json(anno_path)
    videomme_data = []

    for item in data:
        video_file = os.path.join(video_path, item['videoID']+'.mp4')
        type = item['duration']
        question = item['question']
        options = item['options']
        answer = item['answer']

        instruction = 'Question: ' + question + '\nOptions:'
        for i in range(len(options)):
            option = options[i]
            instruction += "\n" + option
        instruction += "\nPlease respond with only the letter of the correct answer."

        videomme_data.append(
            {
                'video_file': video_file,
                'type': type,
                'question': instruction,
                'answer': answer,
            }
        )

    import random
    random.shuffle(videomme_data)
    if 'STREAMBRIDGE_LIMIT_RECORDS' in os.environ:
        videomme_data = videomme_data[: int(os.environ['STREAMBRIDGE_LIMIT_RECORDS'])]
    if 'STREAMBRIDGE_NUM_SHARDS' in os.environ:
        num_shards = int(os.environ['STREAMBRIDGE_NUM_SHARDS'])
        shard_index = int(os.environ.get('STREAMBRIDGE_SHARD_INDEX', 0))
        if not 0 <= shard_index < num_shards:
            raise ValueError("STREAMBRIDGE_SHARD_INDEX must be in [0, STREAMBRIDGE_NUM_SHARDS)")
        videomme_data = [item for index, item in enumerate(videomme_data) if index % num_shards == shard_index]

    if 'NUM_GPUS' in os.environ:
        num_gpus = int(os.environ['NUM_GPUS'])
    else:
        num_gpus = 8
    splits = np.array_split(videomme_data, num_gpus)

    mp.set_start_method('spawn') 
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=evaluate, args=(rank, splits[rank].tolist()))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_res = []
    for rank in range(num_gpus):
        file_path = result_path(f'videomme_eval_gpu{rank}.json')
        res_part = load_json(file_path)
        final_res.extend(res_part)
        os.remove(file_path) 

    save_json(final_res, result_path('videomme_eval.json'))
