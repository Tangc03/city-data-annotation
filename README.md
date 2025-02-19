# City Data Annotation

## Files Structure

`./pretrained` structure:

```
.
├── InternVL2_5-78B
├── Llama-3.3-70B-Instruct
├── Qwen2.5-72B-Instruct
└── Sa2VA-26B
```

`./datasets/city_standard_data` structure:

```
.
├── annotations
├── images
└── images_with_bbox
```

- `internvl.py`: Code for generation via InternVL 2.5-78B
- `qwen.py`: Code for generation via Qwen 2.5-72B-Instruct
- `run.sh`: A script to run two python files 

## Prompts

### Image Description:

For InternVL:

```python
question1 = '<image>\nPlease describe this image in detail. Avoid excessively long and detailed expressions, condense them into a single paragraph without describing the words on the surveillance screen (date, location, etc.) and without describing the words that appear on the screen (content of the sign, etc.)'
```

### QA Generation:

For InternVL:

```python
question2 = '<image>\n' + response1 + '\nGenerate some questions about the image content and give the answer. Avoid describing the words on the surveillance screen (date, location, etc.) and the words that appear on the screen (content of the sign, etc.)'
```

For Qwen:

```python
def process_qa(model, tokenizer, qa_text):
    prompt = f"""
    Extract question-answer pairs from the following text and format them as JSON:
    {qa_text}
    
    Format each pair as:
    {{"question": "...", "answer": "..."}}
    """
    
    response = get_model_response(model, tokenizer, prompt)
    # Parse the response to get structured QA pairs
    qa_pairs = []
    try:
        # Extract all question-answer pairs using regex
        matches = re.findall(r'"question":\s*"([^"]+)"[^}]+?"answer":\s*"([^"]+)"', response)
        for q, a in matches:
            qa_pairs.append({"question": q, "answer": a})
    except:
        print("Error parsing QA response")
    
    return qa_pairs
```

### Region Description:

For InternVL:

```python
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
```

For Qwen:

```python
def simplify_region_description(model, tokenizer, desc):
    prompt = f"""
    Simplify this region description by:
    1. Removing phrases like "the object in the red box"
    2. Only keeping descriptions of the specific region
    3. Making it concise
    
    Original: {desc}
    """
    
    return get_model_response(model, tokenizer, prompt)
```

### Grounded Caption:

For Qwen:

```python
def generate_grounded_caption(model, tokenizer, image_desc, region_descs):
    prompt = f"""
    Generate a comprehensive caption that:
    1. Integrates the main image description: {image_desc}
    2. Incorporates these region descriptions: {region_descs}
    3. Uses special tags like <1>object1</1>, <2>object2</2> to mark specific items
    4. Focuses on events and key objects
    5. Avoid describing the words on the surveillance screen (date, location, etc.) and the words that appear on the screen (content of the sign, etc.)
    6. Avoid some irrelevant part like "**Caption:**"
    """
    
    return get_model_response(model, tokenizer, prompt)
```