import os
from hpsv3 import HPSv3RewardInferencer

# Initialize the model
inferencer = HPSv3RewardInferencer(device='cuda', checkpoint_path=os.environ.get("HPSV3_CKPT", "checkpoints/HPSv3.safetensors"))

# Evaluate images
image_paths = ["assets/example1.png", "assets/example2.png"]
prompts = [
  "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
  "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker"
]

# Get preference scores
rewards = inferencer.reward(prompts, image_paths=image_paths)
scores = [reward[0].item() for reward in rewards]  # Extract mu values
print(f"Image scores: {scores}")