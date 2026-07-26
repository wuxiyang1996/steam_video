#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import logging
import logging.handlers
import os
import sys

import requests
import torch.distributed as dist

from streambridge.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def rank0_print(*args):
    if is_local_leader:
        print(*args)


def is_torch_distributed():
    return dist.is_available() and dist.is_initialized()


def is_local_leader():
    if is_torch_distributed():
        return int(os.environ["LOCAL_RANK"]) == 0
    return True


def dist_barrier():
    if is_torch_distributed():
        dist.barrier()


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


import requests
from PIL import Image
from io import BytesIO
import json
import pickle
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop, ToPILImage, RandomHorizontalFlip, Lambda
from typing import Optional, Tuple, Any, Union, List


def _convert_to_rgb(image):
    return image.convert('RGB')

SIGLIP_DATASET_MEAN = (0.5, 0.5, 0.5)
SIGLIP_DATASET_STD = (0.5, 0.5, 0.5)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

INTERNVIDEO_MEAN = (0.485, 0.456, 0.406)
INTERNVIDEO_STD = (0.229, 0.224, 0.225)

def image_transform(
        image_size: int,
        rescale_factor: float = 1.0,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        random_flip = False,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
    ]

    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def frame_transform(
        image_size: Union[int, Tuple[int, int]],
        rescale_factor: float = 1.0,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        random_flip = False,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, int):
        resize_size = (image_size, image_size)
        crop_size = (image_size, image_size)
    elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        resize_size = (image_size[0], image_size[1])
        crop_size = (image_size[0], image_size[1])
    else:
        raise ValueError("image_size must be an int or a tuple of two ints.")

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        ToPILImage(),
        Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resize_size),
        RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def expand2square(pil_img, background_color=tuple(int(x*255) for x in OPENAI_DATASET_MEAN)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize_and_pad(image_path, target_size):
    if image_path is str:
        with Image.open(image_path) as img:
            # 确保图片是 RGB 格式
            img = img.convert("RGB")
    else:
        img = image_path
    # 调整大小，保持比例
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    # 创建一个黑色背景的图片，大小为 target_size
    padded_img = Image.new("RGB", target_size, (0, 0, 0))
    # 将调整后的图片粘贴到背景中央
    paste_position = (
        (target_size[0] - img.width) // 2,
        (target_size[1] - img.height) // 2
    )
    padded_img.paste(img, paste_position)
    return padded_img

def resize_and_center_crop(image_path, target_size):
    if image_path is str:
        with Image.open(image_path) as img:
            # 确保图片是 RGB 格式
            img = img.convert("RGB")
    else:
        img = image_path
    # 获取原始宽高
    original_width, original_height = img.size
    target_width, target_height = target_size

    # 计算调整大小时的比例
    scale = max(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # 计算中心裁剪的区域
    left = (new_size[0] - target_width) // 2
    top = (new_size[1] - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # 裁剪出中心区域
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img
        
def load_image(image_file, pad=False):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    if pad:
        image = expand2square(image)
    return image

def load_txt(path):
    strings_list = []
    with open(path, 'r') as file:
        for line in file:
            # 去除每行的换行符，并将其添加到列表中
            strings_list.append(line.strip())
    return strings_list

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)
        
def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_csv(path):
    import pandas as pd

    file_list = []
    data = pd.read_csv(path)
    columns = data.columns.tolist()
    for index, row in data.iterrows():
        file_list.append({})
        for column in columns:
            file_list[index][column] = row[column]
    return file_list
