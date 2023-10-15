from pydantic import BaseModel

class StableDiffusionRequest(BaseModel):
  Model: str = "runwayml/stable-diffusion-v1-5"
  Phrase: str
  NegativePhrase: str = None
  Total: int = 1
  Seed: int = None
  GuidanceScale: float = 7.5
  Quality: float = 0.2
  Precision: str = "half" # half, full
  VaeType: str = "original" # original, ema, mse
  AttentionSlicing: bool = False