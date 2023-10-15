from pydantic import BaseModel

class StableDiffusionInpaintRequest(BaseModel):
  Model: str = "runwayml/stable-diffusion-inpainting"
  Phrase: str
  NegativePhrase: str = None
  ImageBase64: str
  MaskImageBase64: str
  Total: int = 1
  Seed: int = None
  GuidanceScale: float = 7.5
  Quality: float = 0.2
  Precision: str = "half" # half, full
  VaeType: str = "original" # original, ema, mse
  AttentionSlicing: bool = False