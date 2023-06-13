import gradio as gr
import torch
import openai
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import os
from safetensors.torch import load_file

# OpenAI api KEY
openai.api_key = 'Your OpenAI API key'

# load .safetensors
def load_lora_weights(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 1
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
            

    return pipeline

    

# Story Generation
def generate_story_openai_turbo(keywords, style, role='children literature writer'):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a {role}."},
            {"role": "user", "content": f"Please write a story below 40 tokens in English based on the following keywords and style, style: {style}, keywords: {', '.join(keywords)}."},
        ],
        max_tokens=3000,
        temperature=0.7,
    )
    return response.choices[0].message['content']

def generate_story_openai_davinci_003(keywords, style, role='children literature writer'):
    # Create a prompt
    #prompt = f"As a {role}, Please write a story of around 200 words in Traditional Chinese ,Story condensed into 50 tokens in English based on the following keywords and style, style: {style}, keywords: {', '.join(keywords)}."
    prompt = f"As a {role}, Please write a story below 40 tokens in English based on the following keywords and style, style: {style}, keywords: {', '.join(keywords)}."
    
    # Generate story with "text-davinci-003"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=3000,
        temperature=0.7
    )

    return response.choices[0].text.strip()

def generate_story_openai_davinci_002(keywords, style, role='children literature writer'):
    # Create a prompt
    # prompt = f"As a {role}, Please write a story of around 200 words in Traditional Chinese ,Story condensed into 50 tokens in English based on the following keywords and style, style: {style}, keywords: {', '.join(keywords)}."
    prompt = f"As a {role}, Please write a story below 40 tokens in English based on the following keywords and style, style: {style}, keywords: {', '.join(keywords)}."
    # Generate story with "text-davinci-002"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=3000,
        temperature=0.7
    )

    return response.choices[0].text.strip()

def generate_story(keywords, style, model_choice):
    if model_choice == 'OpenAI GPT-3.5 Turbo':
        return generate_story_openai_turbo(keywords, style)
    elif model_choice == 'OpenAI text-davinci-003':
        return generate_story_openai_davinci_003(keywords, style)
    elif model_choice == 'OpenAI text-davinci-002':
        return generate_story_openai_davinci_002(keywords, style)

# inputstorytext = gr.Textbox(label="Please enter your story generation prompt")
# outputstorytext = gr.Textbox(label="Story result")

iface1 = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.components.Textbox(lines=2, label='Keywords, separated by commas'), 
        gr.components.Radio(['Adventure', 'Happy', 'Sad', 'Warm'], label='Style'),
        gr.components.Radio(['OpenAI GPT-3.5 Turbo', 'OpenAI text-davinci-003', 'OpenAI text-davinci-002'], label='Model Choice'),

    ],
    outputs=gr.Textbox(label="Story content"),
    title="Story generation",
    description="Please enter some keywords, select a style, and a model to make a story!"
    )

# Illustration Generation
device = "cuda"

# model Counterfeit with lora
def Counterfeit(prompt):
    model_Counterfeit = "gsdf/Counterfeit-V2.5"
    lora_for_Counterfeit = './lora files/gsdf_Counterfeit-V2.5/'
    pipe_Counterfeit = StableDiffusionPipeline.from_pretrained(model_Counterfeit, torch_dtype=torch.float16)
    pipe_Counterfeit.unet.load_attn_procs(lora_for_Counterfeit)
    pipe_Counterfeit = pipe_Counterfeit.to(device)
    image = pipe_Counterfeit(prompt).images[0]
    
    return image
    

# model stable-diffusion-v1-4 with Children's Drawing LoRA
def stable_lora(prompt):
    model_stable = "CompVis/stable-diffusion-v1-4"
    lora_for_stable = './lora files/child-drawing.safetensors'
    pipe_stable = StableDiffusionPipeline.from_pretrained(model_stable, torch_dtype=torch.float16)
    load_lora_weights(pipe_stable, lora_for_stable)
    pipe_stable = pipe_stable.to(device)
    image = pipe_stable(prompt).images[0]
    
    return image

# model stable-diffusion-v1-4
def stable(prompt):
    model_stable_org = 'CompVis/stable-diffusion-v1-4'
    pipe_stable_org = StableDiffusionPipeline.from_pretrained(model_stable_org , torch_dtype=torch.float16)
    pipe_stable_orgn = pipe_stable_org.to(device)
    image = pipe_stable_org(prompt).images[0]
    
    return image


def generate_illustration(story_content, model_choice):
    if model_choice == 'Counterfeit-V2.5 add pokemon LoRA':
        illustration = Counterfeit(story_content)
    elif model_choice == "stable-diffusion-v1-4 add Children's Drawing LoRA":
        illustration = stable_lora(story_content)
    elif model_choice == "stable-diffusion-v1-4":
        illustration = stable(story_content)
        
    torch.cuda.empty_cache()
    
    return illustration


iface2 = gr.Interface(
    fn=generate_illustration,
    inputs=[
        gr.components.Textbox(lines=2, label='Story content'), 
        gr.components.Radio(['Counterfeit-V2.5 add pokemon LoRA', "stable-diffusion-v1-4 add Children's Drawing LoRA", "stable-diffusion-v1-4"], label='Model Choice')
    ],
    outputs=gr.Image(label="Story illustration"),
    title="Illustration generation",
    description="Please enter a story to make an illustration!"
    )

def generate_storyandillustration(keywords, style, model_choice_story, model_choice_illustration):
    story_content = generate_story(keywords, style, model_choice_story)
    
    return story_content, generate_illustration(story_content, model_choice_illustration)


with gr.Blocks() as whole:
    with gr.Row():
        with gr.Column():
            input1 = gr.components.Textbox(lines=2, label='Keywords, separated by commas')
            input2 = gr.components.Radio(['Adventure', 'Happy', 'Sad', 'Warm'], label='Style')
            input3 = gr.components.Radio(['OpenAI GPT-3.5 Turbo', 'OpenAI text-davinci-003', 'OpenAI text-davinci-002'], label='Model Choice for Story')
            input4 =  gr.components.Radio(['Counterfeit-V2.5 add pokemon LoRA', "stable-diffusion-v1-4 add Children's Drawing LoRA", "stable-diffusion-v1-4"], label='Model Choice for Illustration')
        with gr.Column():
            outputstory = gr.Textbox(label="Story content")
            outputillustration = gr.Image(label="Story illustration")
    btn = gr.Button("Generate")
    btn.click(generate_storyandillustration, 
              inputs=[
                        input1,
                        input2,
                        input3,
                        input4,
                    ],
              outputs=[outputstory, outputillustration])

if __name__ == "__main__":
    print(gr.TabbedInterface([iface1, iface2, whole],['Text-to-Story', 'Story-to-Illustration', 'Text-to-Story and Illustration']).launch(share=True))

