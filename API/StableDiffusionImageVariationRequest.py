from pydantic import BaseModel

class StableDiffusionImageVariationRequest(BaseModel):
  ImageBase64: str
  Total: int = 1
  Seed: int = None
  GuidanceScale: float = 3.0
  Quality: float = 0.2
  VaeType: str = "original" # original, ema, mse
  AttentionSlicing: bool = False