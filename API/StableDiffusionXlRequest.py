from pydantic import BaseModel

class StableDiffusionXlRequest(BaseModel):
  Model: str = "stabilityai/stable-diffusion-xl-base-1.0"
  Phrase: str
  NegativePhrase: str = None
  Total: int = 1
  Seed: int = None
  GuidanceScale: float = 5.0
  Quality: float = 0.2
  Precision: str = "half" # half, full
  VaeType: str = "original" # original, ema, mse
  AttentionSlicing: bool = False