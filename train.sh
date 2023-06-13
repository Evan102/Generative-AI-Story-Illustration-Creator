python3 train_text_to_image_lora.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --dataset_name="lambdalabs/pokemon-blip-captions" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --seed=42 \
  --num_train_epochs=5 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model-lora/CompVis_stable-diffusion-v1-4_Fox Xiaohua" \
  --validation_prompt="Fox Xiaohua and Turtle Aman had a race in the forest. Xiaohua ran fast but was overconfident and took a nap. Aman persisted and eventually won."