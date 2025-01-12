import onnxruntime as ort
from transformers import ViTImageProcessor
from PIL import Image
import numpy as np

ort_session = ort.InferenceSession("model_repository_2/vit_model_onnx/1/model.onnx")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="np")
    return inputs["pixel_values"]  # [1, 3, 224, 224]


def predict(image_path):

    input_tensor = preprocess_image(image_path)

    outputs = ort_session.run(None, {"input": input_tensor})

    logits = outputs[0]
    probabilities = softmax(logits[0])

    predicted_class = np.argmax(probabilities)
    class_labels = {0: "cat", 1: "not-cat"}

    return {
        "probabilities": probabilities.tolist(),
        "predicted_class": class_labels[predicted_class]
    }


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


if __name__ == "__main__":
    image_path = "images/dog_2.png"
    result = predict(image_path)
    print(f"Sonuç: {result}")
