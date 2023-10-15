from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionImageVariationPipeline
from diffusers import StableDiffusionInpaintPipeline
from torch import torch
from torchvision import transforms
from traitlets import Callable
from ArgTransforms import ArgTransforms
from ImageLayer import ImageLayer
from PipelineOptions import PipelineOptions
from Utility import Utility
from API.StableDiffusionRequest import StableDiffusionRequest
from API.StableDiffusion2Request import StableDiffusion2Request
from API.StableDiffusionXlRequest import StableDiffusionXlRequest
from API.StableDiffusionImageToImageRequest import StableDiffusionImageToImageRequest
from API.StableDiffusionImageVariationRequest import StableDiffusionImageVariationRequest
from API.StableDiffusionInpaintRequest import StableDiffusionInpaintRequest

class StableDiffusion:
  def _InternalGenerate(name: str, request, pipeline_options: PipelineOptions, create_pipeline: Callable):
    torch_dtype = ArgTransforms.GetPrecision(request.Precision)
    pipeline = create_pipeline(request.VaeType, request.Model, torch_dtype)

    if pipeline_options.IsXformersEnabled:
      pipeline.enable_xformers_memory_efficient_attention()

    if request.AttentionSlicing:
      pipeline.enable_attention_slicing()

    pipeline.enable_model_cpu_offload()

    seed = ArgTransforms.GetSeed(request.Seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    steps_total = ArgTransforms.GetQuality(request.Quality)
    images_total = ArgTransforms.GetTotal(request.Total)
    guidance_scale = ArgTransforms.GetGuidanceScale(request.GuidanceScale)

    images = pipeline(
      request.Phrase,
      num_inference_steps=steps_total,
      num_images_per_prompt=images_total,
      guidance_scale=guidance_scale,
      negative_prompt=request.NegativePhrase,
      generator=generator).images
    results = []

    for image_index, image in enumerate(images):
      results.append(ImageLayer(
          name,
          Utility.GetImageBase64(image),
          {
            "Model": request.Model,
            "Phrase": request.Phrase,
            "NegativePhrase": request.NegativePhrase,
            "Seed": seed,
            "GuidanceScale": guidance_scale,
            "Quality": steps_total,
            "Precision": str(torch_dtype),
            "Index": image_index
          }))

    return results

  def _CreateStableDiffusionPipeline(vae_type: str, model: str, torch_dtype: torch.dtype):
    return StableDiffusionPipeline.from_pretrained(
      model,
      torch_dtype=torch_dtype,
      vae=ArgTransforms.GetVae(vae_type, model, torch_dtype),
      safety_checker=None)
  
  def _CreateStableDiffusion2Pipeline(vae_type: str, model: str, torch_dtype: torch.dtype):
    pipeline = DiffusionPipeline.from_pretrained(
      model,
      torch_dtype=torch_dtype,
      vae=ArgTransforms.GetVae(vae_type, model, torch_dtype),
      safety_checker=None)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline
  
  def _CreateStableDiffusionXlPipeline(vae_type: str, model: str, torch_dtype: torch.dtype):
    return StableDiffusionXLPipeline.from_pretrained(
      model,
      torch_dtype=torch_dtype,
      vae=ArgTransforms.GetVae(vae_type, model, torch_dtype),
      add_watermarker=False,
      use_safetensors=True)

  def Generate(request: StableDiffusionRequest, pipeline_options: PipelineOptions):
    return StableDiffusion._InternalGenerate("StableDiffusion", request, pipeline_options, StableDiffusion._CreateStableDiffusionPipeline)
  
  def Generate2(request: StableDiffusion2Request, pipeline_options: PipelineOptions):
    return StableDiffusion._InternalGenerate("StableDiffusion2", request, pipeline_options, StableDiffusion._CreateStableDiffusion2Pipeline)
  
  def GenerateXl(request: StableDiffusionXlRequest, pipeline_options: PipelineOptions):
    return StableDiffusion._InternalGenerate("StableDiffusionXL", request, pipeline_options, StableDiffusion._CreateStableDiffusionXlPipeline)
  
  def ImageToImage(request: StableDiffusionImageToImageRequest, pipeline_options: PipelineOptions):
    torch_dtype = ArgTransforms.GetPrecision(request.Precision)
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
      request.Model,
      torch_dtype=torch_dtype,
      vae=ArgTransforms.GetVae(request.VaeType, request.Model, torch_dtype),
      safety_checker=None)

    if pipeline_options.IsXformersEnabled:
      pipeline.enable_xformers_memory_efficient_attention()

    if request.AttentionSlicing:
      pipeline.enable_attention_slicing()

    pipeline.enable_model_cpu_offload()

    seed = ArgTransforms.GetSeed(request.Seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    steps_total = ArgTransforms.GetQuality(request.Quality)
    images_total = ArgTransforms.GetTotal(request.Total)
    guidance_scale = ArgTransforms.GetGuidanceScale(request.GuidanceScale)
    strength = ArgTransforms.GetStrength(request.Strength)
    input_image = ArgTransforms.GetImage(request.ImageBase64).convert("RGB")

    images = pipeline(
      prompt=request.Phrase,
      image=input_image,
      strength=strength,
      num_inference_steps=steps_total,
      num_images_per_prompt=images_total,
      guidance_scale=guidance_scale,
      negative_prompt=request.NegativePhrase,
      generator=generator).images
    results = []

    for image_index, image in enumerate(images):
      results.append(ImageLayer(
        "StableDiffusionImageToImage",
        Utility.GetImageBase64(image),
        {
          "Model": request.Model,
          "Phrase": request.Phrase,
          "NegativePhrase": request.NegativePhrase,
          "Strength": strength,
          "Seed": seed,
          "GuidanceScale": guidance_scale,
          "Quality": steps_total,
          "Precision": str(torch_dtype),
          "Index": image_index
        }))

    return results
  
  def ImageVariation(request: StableDiffusionImageVariationRequest, pipeline_options: PipelineOptions):
    pipeline = StableDiffusionImageVariationPipeline.from_pretrained(
      "lambdalabs/sd-image-variations-diffusers",
      revision="v2.0",
      vae=ArgTransforms.GetVae(request.VaeType, request.Model, torch.float16),
      safety_checker=None)

    if pipeline_options.IsXformersEnabled:
      pipeline.enable_xformers_memory_efficient_attention()

    if request.AttentionSlicing:
      pipeline.enable_attention_slicing()

    pipeline.enable_model_cpu_offload()

    seed = ArgTransforms.GetSeed(request.Seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    steps_total = ArgTransforms.GetQuality(request.Quality)
    images_total = ArgTransforms.GetTotal(request.Total)
    guidance_scale = ArgTransforms.GetGuidanceScale(request.GuidanceScale)
    input_image = ArgTransforms.GetImage(request.ImageBase64)
    input_image_transform = transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
      ])
    input_image_transformed = input_image_transform(input_image).to("cuda").unsqueeze(0)

    images = pipeline(
      image=input_image_transformed,
      num_inference_steps=steps_total,
      num_images_per_prompt=images_total,
      guidance_scale=guidance_scale,
      generator=generator).images
    results = []

    for image_index, image in enumerate(images):
      results.append(ImageLayer(
        "StableDiffusionImageVariation",
        Utility.GetImageBase64(image),
        {
          "Seed": seed,
          "GuidanceScale": guidance_scale,
          "Quality": steps_total,
          "Index": image_index
        }))

    return results
  
  def Inpaint(request: StableDiffusionInpaintRequest, pipeline_options: PipelineOptions):
    torch_dtype = ArgTransforms.GetPrecision(request.Precision)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
      request.Model,
      torch_dtype=torch_dtype,
      vae=ArgTransforms.GetVae(request.VaeType, request.Model, torch_dtype),
      safety_checker=None)

    if pipeline_options.IsXformersEnabled:
      pipeline.enable_xformers_memory_efficient_attention()

    if request.AttentionSlicing:
      pipeline.enable_attention_slicing()

    pipeline.enable_model_cpu_offload()

    seed = ArgTransforms.GetSeed(request.Seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    steps_total = ArgTransforms.GetQuality(request.Quality)
    images_total = ArgTransforms.GetTotal(request.Total)
    guidance_scale = ArgTransforms.GetGuidanceScale(request.GuidanceScale)
    input_image = ArgTransforms.GetImage(request.ImageBase64).convert("RGB")
    input_image_mask = ArgTransforms.GetImage(request.MaskImageBase64).convert("RGB")

    images = pipeline(
      prompt=request.Phrase,
      image=input_image,
      mask_image=input_image_mask,
      num_inference_steps=steps_total,
      num_images_per_prompt=images_total,
      guidance_scale=guidance_scale,
      negative_prompt=request.NegativePhrase,
      generator=generator).images
    results = []

    for image_index, image in enumerate(images):
      results.append(ImageLayer(
        "StableDiffusionInpaint",
        Utility.GetImageBase64(image),
        {
          "Model": request.Model,
          "Phrase": request.Phrase,
          "NegativePhrase": request.NegativePhrase,
          "Seed": seed,
          "GuidanceScale": guidance_scale,
          "Quality": steps_total,
          "Precision": str(torch_dtype),
          "Index": image_index
        }))

    return results