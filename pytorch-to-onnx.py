# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "onnx",
#     "rich",
#     "timm",
#     "torch",
# ]
# ///
import torch
import timm
from pathlib import Path
from rich import print
import sys

if len(sys.argv) < 3:
  print("Usage: uv run pytorch-to-onnx.py PATH_TO_PTH PATH_TO_CLASSMAPPING_TXT")
  sys.exit()

filename = Path(sys.argv[1])

classmapping = list(Path(sys.argv[2]).read_text().splitlines())

torch_model = timm.create_model(
    "resnet50.a1_in1k", pretrained=True, num_classes=len(classmapping)
)

state = torch.load(filename, map_location=torch.device("cpu"), weights_only=False)
torch_model.load_state_dict(state["state_dict"])

torch_model = torch.nn.Sequential(torch_model, torch.nn.Softmax(dim=1))
torch_model.eval()

torch.onnx.export(
    torch_model, args=(torch.zeros([1, 3, 224, 224]),), f=filename.with_suffix(".onnx")
)
