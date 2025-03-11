import torch
import pytorch_lightning as pl
import onnx

from unet import LitUnet  

ckpt_path = "trained_unet.ckpt"


model = LitUnet.load_from_checkpoint(ckpt_path)
model.eval() 

example_input = (torch.randn(1, 3, 400, 400))
model.to_onnx('trained_unet_model.onnx', example_input)

onnx_model = onnx.load("trained_unet_model.onnx")
onnx.checker.check_model(onnx_model)