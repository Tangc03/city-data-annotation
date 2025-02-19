import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import tqdm
import argparse

def initialize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="internvl_outputs.json")
    parser.add_argument("--output", type=str, default="qwen_outputs.json")
    return parser

def setup_model():
    model_name = "pretrained/Qwen2.5-72B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_model_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant focused on text processing."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

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

def simplify_region_description(model, tokenizer, desc):
    prompt = f"""
    Simplify this region description by:
    1. Removing phrases like "the object in the red box"
    2. Only keeping descriptions of the specific region
    3. Making it concise
    
    Original: {desc}
    """
    
    return get_model_response(model, tokenizer, prompt)

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

def main():
    parser = initialize_parser()
    model, tokenizer = setup_model()

    input_file = parser.parse_args().input
    output_file = parser.parse_args().output
    
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize output file with empty list
    with open(output_file, 'w') as f:
        f.write('[\n')
    
    for i, item in enumerate(tqdm.tqdm(data)):
        # Process QA sections
        qa_pairs = process_qa(model, tokenizer, item['question_answer'])
        
        # Process region descriptions
        simplified_events = []
        for event in item['bboxes']:
            simplified_desc = simplify_region_description(
                model, 
                tokenizer,
                event['region_description']
            )
            event['region_description'] = simplified_desc
            simplified_events.append(event)
            
        # Generate grounded caption
        grounded_caption = generate_grounded_caption(
            model,
            tokenizer,
            item['image_description'],
            [e['region_description'] for e in simplified_events]
        )
        
        # Create processed item
        processed_item = {
            **item,
            'question_answer': qa_pairs,
            'events': simplified_events,
            'grounded_caption': grounded_caption
        }
        
        # Append to output file
        with open(output_file, 'a') as f:
            json.dump(processed_item, f, indent=2)
            if i < len(data) - 1:  # Add comma for all items except the last
                f.write(',\n')
            else:
                f.write('\n')
    
    # Close the JSON array
    with open(output_file, 'a') as f:
        f.write(']')

if __name__ == "__main__":
    print("Running Qwen script...")
    main()