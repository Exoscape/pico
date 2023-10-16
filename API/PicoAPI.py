from API.PicoResponse import PicoResponse
from API.StableDiffusionRequest import StableDiffusionRequest
from API.StableDiffusion2Request import StableDiffusion2Request
from API.StableDiffusionXlRequest import StableDiffusionXlRequest
from API.StableDiffusionImageToImageRequest import StableDiffusionImageToImageRequest
from API.StableDiffusionImageVariationRequest import StableDiffusionImageVariationRequest
from API.StableDiffusionInpaintRequest import StableDiffusionInpaintRequest
from Models.StableDiffusion import StableDiffusion
from ConcurrencyGate import ConcurrencyGate
from traitlets import Callable
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import traceback

from PipelineOptions import PipelineOptions

class PicoAPI:
  def __init__(self, pipeline_options: PipelineOptions, host: str = "0.0.0.0", port: int = 5000, is_debug: bool = False):
    self._Api = FastAPI()
    self._Api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])
    self._ApiRouter = APIRouter()
    self._GpuGate = ConcurrencyGate(pipeline_options.MaxConcurrency)
    self._PipelineOptions = pipeline_options
    self._Host = host
    self._Port = port
    self._IsDebug = is_debug
    
    self._ApiRouter.add_api_route("/api/v1/health", self._GetHealth, methods=["GET"])

    self._ApiRouter.add_api_route("/api/v1/stable-diffusion", self._InvokeStableDiffusion, methods=["POST"])
    self._ApiRouter.add_api_route("/api/v1/stable-diffusion-2", self._InvokeStableDiffusion2, methods=["POST"])
    self._ApiRouter.add_api_route("/api/v1/stable-diffusion-xl", self._InvokeStableDiffusionXl, methods=["POST"])
    self._ApiRouter.add_api_route("/api/v1/stable-diffusion-image-to-image", self._InvokeStableDiffusionImageToImage, methods=["POST"])
    self._ApiRouter.add_api_route("/api/v1/stable-diffusion-image-variation", self._InvokeStableDiffusionImageVariation, methods=["POST"])
    self._ApiRouter.add_api_route("/api/v1/stable-diffusion-inpaint", self._InvokeStableDiffusionInpaint, methods=["POST"])

    self._Api.include_router(self._ApiRouter)

    print(f"PICO: API listening on {self._Host}:{self._Port}...")
    uvicorn.run(self._Api, host=self._Host, port=self._Port)

  def _InvokePipeline(self, pipeline: Callable, gpu_gate: ConcurrencyGate = None):
    if not gpu_gate is None and not gpu_gate.Acquire():
      raise HTTPException(status_code=503, detail="GPU_BUSY")

    start_time = time.time()
    stop_time = start_time

    try:
      results = pipeline()
      stop_time = time.time()
    except Exception as exception:
      exception_details = exception

      if self._IsDebug:
        exception_details = "".join(traceback.format_exception(None, exception, exception.__traceback__))

      print(f"An exception has occurred: {exception_details}")
      raise HTTPException(status_code=500, detail="INTERNAL_SERVER_ERROR")
    finally:
      if not gpu_gate is None:
        gpu_gate.Release()

    return PicoResponse(
      ElapsedSeconds = stop_time - start_time,
      Images = results)

  def _GetHealth(self):
    return {
      "Status": "Idle" if not self._GpuGate.IsLocked() else "Busy"
    }
  
  def _InvokeStableDiffusion(self, request: StableDiffusionRequest):
    return self._InvokePipeline(lambda: StableDiffusion.Generate(request, self._PipelineOptions), self._GpuGate)
  
  def _InvokeStableDiffusion2(self, request: StableDiffusion2Request):
    return self._InvokePipeline(lambda: StableDiffusion.Generate2(request, self._PipelineOptions), self._GpuGate)
  
  def _InvokeStableDiffusionXl(self, request: StableDiffusionXlRequest):
    return self._InvokePipeline(lambda: StableDiffusion.GenerateXl(request, self._PipelineOptions), self._GpuGate)
  
  def _InvokeStableDiffusionImageToImage(self, request: StableDiffusionImageToImageRequest):
    return self._InvokePipeline(lambda: StableDiffusion.ImageToImage(request, self._PipelineOptions), self._GpuGate)
  
  def _InvokeStableDiffusionImageVariation(self, request: StableDiffusionImageVariationRequest):
    return self._InvokePipeline(lambda: StableDiffusion.ImageVariation(request, self._PipelineOptions), self._GpuGate)
  
  def _InvokeStableDiffusionInpaint(self, request: StableDiffusionInpaintRequest):
    return self._InvokePipeline(lambda: StableDiffusion.Inpaint(request, self._PipelineOptions), self._GpuGate)