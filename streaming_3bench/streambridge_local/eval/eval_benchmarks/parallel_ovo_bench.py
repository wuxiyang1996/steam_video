#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torch.multiprocessing as mp
import glob
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


def resolve_video_path(video_root, video_name, qid=None):
    path = os.path.join(video_root, video_name)
    if os.path.exists(path):
        return path
    if video_name.startswith("data/"):
        stripped = os.path.join(video_root, video_name[len("data/") :])
        if os.path.exists(stripped):
            return stripped
    if qid is not None:
        chunk_root = os.path.join(os.path.dirname(video_root), "chunked_videos")
        matches = sorted(
            glob.glob(os.path.join(chunk_root, f"{qid}_*.mp4")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1])
            if os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1].isdigit()
            else os.path.basename(p),
        )
        if matches:
            return matches[0]
    return path

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
        video_path = resolve_video_path(os.environ['VIDEO_PATH'], item['video'], item.get('qid'))
        
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
        
        frame_intervals, questions, answers, types = [], [], [], []
        options = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']
        start_sec = 0

        for anno in item['anno']:
            if 'options' in anno.keys():
                question = anno['question'] + ' Options:\n' + "\n".join([options[i]+anno['options'][i] for i in range(len(anno['options']))]) + "\nPlease respond with only the letter of the correct answer."
                answer = options[anno['gt']] + anno['answer']
            else:
                question = anno['question']
                answer = anno['answer']
            time = anno['realtime']
            types.append(anno['task'])
            questions.append(question)
            answers.append(answer)
            frame_intervals.append([int(start_sec*sampled_fps), int(time*sampled_fps)])
            start_sec = time

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

        for i in range(len(questions)):
            interval = frame_intervals[i]
            question = questions[i]
            answer = answers[i]
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
                    try:
                        if qwen35_process:
                            for j in range(interval[0], interval[1]):
                                model.receive_one_frame(timestamp_s=(j + 1) / sampled_fps)
                        else:
                            for j in range(interval[0], interval[1]):
                                if j < frame_pixel_values.shape[0]:
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
                'video': item['video'],
                'time': interval[1],
                'type': types[i],
                'question': question,
                'pred': pred_text,
                'answer': answer,
                'model_name': model_name,
                'ckpt': str(ckpt),
                'ok': error_text is None,
                'error': error_text,
                'streaming_visible_until_s': interval[1] / sampled_fps,
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

        save_json(res, result_path(f'ovo_bench_eval_gpu{rank}.json'))

if __name__ == "__main__":

    anno_path = os.environ['ANNO_PATH']
    data = load_json(anno_path)
    import random
    video_dic = {}
    target_tasks = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR', 'EPM', 'ASI', 'HLD']

    for item in data:
        video = item['video']
        task = item['task']
        if task in target_tasks:
            video_dic.setdefault(video, []).append({k: v for k, v in item.items() if k != "video"})

    stream_data = [
        {'video': key, 'qid': str(anno[0].get("id")) if anno and anno[0].get("id") is not None else None, 'anno': sorted(anno, key=lambda x: x["realtime"])}
        for key, anno in video_dic.items()
    ]

    FAR_data = []
    far_tasks = ['REC', 'SSR', 'CRR']
    for item in data:
        video = item['video']
        task = item['task']
        if task in far_tasks:
            test_info = item['test_info']
            sample = {
                'video': video,
                'qid': str(item.get('id')) if item.get('id') is not None else None,
                'task': task,
                'anno': [],
            }
            if task == 'REC':
                for info in test_info:
                    prompt = "Your task is to count how many times did different people in the video perform some kind of action in total. You should directly response the count in an INT format only, for example, 0/1/2/3..., without any other information.\n"
                    prompt = ''
                    question = prompt + f"How many times did the action '{item['activity']}' happened? Only answer me an Arabic Numerals with INT format only, for example, 0/1/2/3..."
                    answer = str(info['count'])
                    sample['anno'].append({
                        'id': item['id'],
                        'task': task,
                        'realtime': info['realtime'],
                        'question': question,
                        'answer': answer,
                    })
            elif task == 'SSR':
                for info in test_info:
                    prompt = "You're watching a tutorial video which contain a sequential of steps. The following is one step from the whole procedures, Your task is to decide: Is the man/woman in the video currently carrying out this step?. Return 'Yes' only if the man/woman in the video is currently performing this step; Return 'No' only if not.\n"
                    prompt = ''
                    question = prompt + f"Is the man/woman in the video currently carrying out the step: '{info['step']}'? Only answer me yes or no."
                    answer = 'No' if info['type'] == 0 else 'Yes'
                    sample['anno'].append({
                        'id': item['id'],
                        'task': task,
                        'realtime': info['realtime'],
                        'question': question,
                        'answer': answer,
                    })
            elif task == 'CRR':
                for info in test_info:
                    prompt = "You're responsible of answering questions based on the video content. The following question are relevant to the latest frames, i.e. the end of the video. Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question. Return 'Yes' only if existing visual content has provided enough information; Return 'No' only otherwise.\n"
                    raw_question = item['question'].replace("?",".")
                    prompt = ''
                    question = prompt + f"Is the current information enough to answer the question: '{raw_question}'? Only answer me yes or no."
                    answer = 'No' if info['type'] == 0 else 'Yes'
                    sample['anno'].append({
                        'id': item['id'],
                        'task': task,
                        'realtime': info['realtime'],
                        'question': question,
                        'answer': answer,
                    })
            unique_annos = {dic['realtime']: dic for dic in sample['anno']}.values()
            sorted_annos = sorted(unique_annos, key=lambda x: x['realtime'])
            sample['anno'] = sorted_annos
            FAR_data.append(sample)

    stream_data += FAR_data
    random.shuffle(stream_data)
    if 'STREAMBRIDGE_LIMIT_VIDEOS' in os.environ:
        stream_data = stream_data[: int(os.environ['STREAMBRIDGE_LIMIT_VIDEOS'])]
    if 'STREAMBRIDGE_NUM_SHARDS' in os.environ:
        num_shards = int(os.environ['STREAMBRIDGE_NUM_SHARDS'])
        shard_index = int(os.environ.get('STREAMBRIDGE_SHARD_INDEX', 0))
        if not 0 <= shard_index < num_shards:
            raise ValueError("STREAMBRIDGE_SHARD_INDEX must be in [0, STREAMBRIDGE_NUM_SHARDS)")
        stream_data = [item for index, item in enumerate(stream_data) if index % num_shards == shard_index]

    if 'NUM_GPUS' in os.environ:
        num_gpus = int(os.environ['NUM_GPUS'])
    else:
        num_gpus = 8
    splits = np.array_split(stream_data, num_gpus)

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
        file_path = result_path(f'ovo_bench_eval_gpu{rank}.json')
        res_part = load_json(file_path)
        final_res.extend(res_part)
        os.remove(file_path)

    save_json(final_res, result_path('ovo_bench_eval.json'))
