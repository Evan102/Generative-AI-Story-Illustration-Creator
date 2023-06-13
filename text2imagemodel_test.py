import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import os
from safetensors.torch import load_file

# load .safetensors
def load_lora_weights(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
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

# model type
model_id = "CompVis/stable-diffusion-v1-4"
lora_model_path = './lora files/CompVis_stable-diffusion-v1-4/'
device = "cuda"

dirs = './CompVis_stable-diffusion-v1-4/lora_test/'
os.makedirs(dirs, exist_ok=True)

# load model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# load lora weight
pipe.unet.load_attn_procs(lora_model_path)
# load_lora_weights(pipe, lora_model_path) #for .safetensors files

pipe = pipe.to(device)

pipe.safety_checker = lambda images, clip_input: (images, False)

# Prompts
frontprompt = 'manga, details line without color style: '
story1 = "Fox Xiaohua and Turtle Aman had a race in the forest. Xiaohua ran fast but was overconfident and took a nap. Aman persisted and eventually won."
story2 = "Monkey Aqiang was playing in a tree when he spotted a rainbow on the ground. He slid down and admired the beautiful rainbow with tiger Eva, forming friendships."
story3 = "Bunny Jiaojiao met bird Anna during the Moon Festival. It taught Jiaojiao how to make delicious pastries with moonlight, and they became good friends, sharing happy moments."
backprompt = '--niji'


# settings
chosenstory = story1
storyname ='lora_Fox Xiaohua'

# prompt1
formodelprompt1 = chosenstory
print('formodelprompt1: ',formodelprompt1)
image1 = pipe(formodelprompt1).images[0]  
image1.save(dirs+storyname+"1.png")

# prompt2
formodelprompt2 = frontprompt+chosenstory
print('formodelprompt2: ',formodelprompt2)
image2 = pipe(formodelprompt2).images[0]  
image2.save(dirs+storyname+"2.png")

# prompt3
formodelprompt3 = frontprompt+chosenstory+backprompt 
print('formodelprompt3: ',formodelprompt3)
image3 = pipe(formodelprompt3).images[0]  
image3.save(dirs+storyname+"3.png")

