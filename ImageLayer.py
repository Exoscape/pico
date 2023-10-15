import base64
from io import BytesIO
from typing import List
from PIL import Image

class ImageLayer:
  def __init__(self, name: str, image: str, attributes: dict = {}):
    self.Name = name
    self.Image = image
    self.Attributes = attributes

  def FromDisk(name: str, path: str):
    image_buffer = open(path, "rb")
    return ImageLayer(name, base64.b64encode(image_buffer.read()).decode("utf-8"))

  def Save(self, path: str):
    image_bytes = base64.b64decode(self.Image)
    image_buffer = BytesIO(image_bytes)
    with open(path, "wb") as image_file:
      image_file.write(image_buffer.getbuffer())

  def AsBytes(self):
    return base64.b64decode(self.Image)

  def AsImage(self):
    image_bytes = BytesIO(self.AsBytes())
    image = Image.open(image_bytes)
    return image

  def GetAttribute(self, attribute_name: str):
    return self.Attributes[attribute_name]

  def SetAttribute(self, attribute_name: str, attribute_value):
    self.Attributes[attribute_name] = attribute_value

  def GetDistinct(layers: List["ImageLayer"], attribute_name: str):
    distinct = []
    seen = set()
    
    for layer in layers:
      attribute_value = layer.GetAttribute(attribute_name)

      if not attribute_value in seen:
        seen.add(attribute_value)
        distinct.append(attribute_value)

    return distinct