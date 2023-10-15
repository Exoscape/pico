from pydantic import BaseModel
from ImageLayer import ImageLayer

class PicoResponse(BaseModel):
  ElapsedSeconds: float
  Images: list[ImageLayer]

  class Config:
    arbitrary_types_allowed = True