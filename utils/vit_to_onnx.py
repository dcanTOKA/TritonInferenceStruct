from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
import onnx
import torch.onnx

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    id2label={0: "cat", 1: "not-cat"},
    label2id={"cat": 0, "not-cat": 1},
    ignore_mismatched_sizes=True
)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
dummy_image = Image.fromarray(dummy_image)

inputs = processor(images=dummy_image, return_tensors="pt")["pixel_values"]  # [1, 3, 224, 224]

torch.onnx.export(
    model,
    args=(inputs,),
    f="model.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX modeli başarıyla oluşturuldu: model.onnx")
