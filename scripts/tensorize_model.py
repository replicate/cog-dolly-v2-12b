#!/usr/bin/env python
import torch

from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model = AutoModelForCausalLM.from_pretrained("./pretrained_weights/", torch_dtype=torch.float16).to('cuda:0')

path = './tensorized_models/dolly-v2-12b-fp16.tensors'
serializer = TensorSerializer(path)
serializer.write_module(model)
serializer.close()