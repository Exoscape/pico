import base64
from PIL import Image
from io import BytesIO

class Utility:
  def GetImageBase64(image: Image):
    image_buffer = BytesIO()
    image.save(image_buffer, format="png")
    return base64.b64encode(image_buffer.getvalue()).decode("utf-8")
  
  def GetImageFromBase64(image_base64: str):
    image_bytes = BytesIO(base64.b64decode(image_base64))
    image = Image.open(image_bytes)
    return image

  def Clamp(value, min_value, max_value, min_clamp=None, max_clamp=None):
    if min_clamp is None:
      min_clamp = min_value
    if max_clamp is None:
      max_clamp = max_value
    if value is None:
      return value
    if value < min_value:
      return min_clamp
    elif value > max_value:
      return max_clamp

    return value