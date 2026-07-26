#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
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
            img = img.convert("RGB")
    else:
        img = image_path
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    padded_img = Image.new("RGB", target_size, (0, 0, 0))
    paste_position = (
        (target_size[0] - img.width) // 2,
        (target_size[1] - img.height) // 2
    )
    padded_img.paste(img, paste_position)
    return padded_img

def resize_and_center_crop(image_path, target_size):
    if image_path is str:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
    else:
        img = image_path
    original_width, original_height = img.size
    target_width, target_height = target_size

    scale = max(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    left = (new_size[0] - target_width) // 2
    top = (new_size[1] - target_height) // 2
    right = left + target_width
    bottom = top + target_height

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

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 
