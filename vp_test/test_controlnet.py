from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained("mattyamonaca/controlnet_vp_one_proto", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "852wa/SDHK", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

control_image = load_image("../flagged/image.png") #入力したいテスト画像のPATHを入力してください
prompt = "high quality, simple, single point perspective, one point perspective, anime, roadway, pathway"
negative_prompt = "eople, greyscale, ugly, deformed, noisy, low-contrast, worst quality, two point perspective, railroad, rail, "
generator = torch.manual_seed(10)
image = pipeline(
    prompt,
    negative_prompt = negative_prompt,
    controlnet_conditioning_scale = 0.8,
    controlnet_conditioning_scalenum_inference_steps=20,
    generator=generator,
    image=control_image).images[0]
image.save("./output.png")