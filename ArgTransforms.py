import random
from torch import torch
from Utility import Utility
from diffusers import AutoencoderKL

class ArgTransforms:
  def GetPrecision(value: str):
    return torch.float16 if value == "half" else torch.float32
  
  def GetVae(value: str, model: str, torch_dtype: torch.dtype):
    match value:
      case "ema":
        return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch_dtype).to("cuda")
      case "mse":
        return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch_dtype).to("cuda")
      case _:
        return AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=torch_dtype).to("cuda")
  
  def GetSeed(value: int):
    seed_max = 2**32 - 1

    if value is None:
      value = random.randint(0, seed_max)

    return Utility.Clamp(value, 0, seed_max)
  
  def GetQuality(value: float):
    quality_max = 250
    quality = Utility.Clamp(value, 1 / quality_max, 1.0)
    return round(Utility.Clamp(quality_max * quality, 1, quality_max))
  
  def GetTotal(value: int):
    return Utility.Clamp(value, 1, 5)
  
  def GetGuidanceScale(value: float):
    return Utility.Clamp(value, 0.0, 10.0)
  
  def GetStrength(value: float):
    return Utility.Clamp(value, 0.0, 1.0)
  
  def GetImage(value: str):
    return Utility.GetImageFromBase64(value)