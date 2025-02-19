import os
import torch
from transformers import AutoTokenizer, AutoModel

import math

import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode

import json
from PIL import Image, ImageDraw
import argparse
import tqdm

def initialize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="internvl_outputs.json")
    return parser

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 32, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 60, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def main():
    parser = initialize_parser()
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    path = 'pretrained/InternVL2_5-78B'
    device_map = split_model('InternVL2_5-78B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    root = "datasets/city_standard_data"
    data_root = os.path.join(root, "images")
    annotation_root = os.path.join(root, "annotations")
    json_file = os.path.join(annotation_root, "annotations.json")

    output_file = parser.parse_args().output

    # 首先读取已有的输出文件(如果存在)
    processed_examples = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8',errors='ignore') as f:
            processed_examples = {ex['image_id']: ex for ex in json.load(f)}
    
    with open(json_file, 'r', encoding='utf-8',errors='ignore') as f:
        examples = json.load(f)

    # 加进度条
    for example in tqdm.tqdm(examples):
        image_id = example['image_id']

        # 如果该example已经处理过，则跳过
        if image_id in processed_examples:
            continue

        image_file = os.path.join(data_root, f"{image_id}.jpg")
        pixel_values = load_image(image_file, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        # single-image single-round conversation (单图单轮对话)
        question1 = '<image>\nPlease describe this image in detail. Avoid excessively long and detailed expressions, condense them into a single paragraph without describing the words on the surveillance screen (date, location, etc.) and without describing the words that appear on the screen (content of the sign, etc.)'
        response1 = model.chat(tokenizer, pixel_values, question1, generation_config)
        # print("Image:", image_file)
        # print("Response1:", response1)

        # question2 = image + response + 'Generate some questions about the image content and give the answer'
        question2 = '<image>\n' + response1 + '\nGenerate some questions about the image content and give the answer. Avoid describing the words on the surveillance screen (date, location, etc.) and the words that appear on the screen (content of the sign, etc.)'
        response2 = model.chat(tokenizer, pixel_values, question2, generation_config)
        # print("Image:", image_file)
        # print("Response2:", response2)

        example['image_description'] = response1
        example['question_answer'] = response2

        for event in tqdm.tqdm(example['bboxes']):
            bbox = event['bbox']
            region_image = Image.open(image_file).convert('RGB')

            # 在原图上画出bbox，在不保存不展示的情况下继续完成chat
            draw = ImageDraw.Draw(region_image)
            draw.rectangle(bbox, outline='red')

            # 对region_image进行chat，不用截取bbox，直接使用整张图片
            transform = build_transform(input_size=448)
            dynamic_images = dynamic_preprocess(region_image, image_size=448, use_thumbnail=True, max_num=12)
            region_pixel_values = [transform(image) for image in dynamic_images]
            region_pixel_values = torch.stack(region_pixel_values)
            region_pixel_values = region_pixel_values.to(torch.bfloat16).cuda()

            if event['is_event'] == 0:
                # 简要描述物体的位置和特征
                item_type = event['type']
                question3 = '<image>\n The object in the red box is ' + item_type + '. Please describe the location and characteristics of the object briefly.'
            elif event['is_event'] == 1:
                # 给出该事件的定义，要求MLLM详细描述该事件的内容
                event_type = event['type']
                question3 = '<image>\n The event happening in the red box is called ' + event_type + '. Please describe the content of the event in detail.'

            response3 = model.chat(tokenizer, region_pixel_values, question3, generation_config)
            # print("Response3:", response3)
            event['region_description'] = response3

        # 每处理完一个example就更新并写入文件，注意写入内容中有中文，需要指定编码格式
        processed_examples[image_id] = example
        with open(output_file, 'w', encoding='utf-8',errors='ignore') as f:
            json.dump(list(processed_examples.values()), f, indent=4)

        # processed_examples[image_id] = example
        # with open(output_file, 'w') as f:
        #     json.dump(list(processed_examples.values()), f, indent=4)

        # json_str = json.dumps(dict,ensure_ascii=False,indent=4)
        # with open('class.json', 'w',encoding='utf-8') as json_file:
        #     json_file.write(json_str)


    # output_file = parser.parse_args().output
    # with open(output_file, 'w') as f:
    #     json.dump(examples, f, indent=4)

if __name__ == '__main__':
    print("Running InternVL script...")
    main()

# running command:
# python internvl.py --output internvl_outputs.json