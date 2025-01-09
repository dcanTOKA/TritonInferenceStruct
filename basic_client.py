from typing import List

import numpy as np
import tritonclient.http as httpclient

import time


class TritonClient:
    def __init__(self, server_url: str, model_name: str):
        self.server_url = server_url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=server_url)

    @staticmethod
    def preprocess_image(image_path: str):
        with open(image_path, "rb") as f:
            img = np.frombuffer(f.read(), dtype=np.uint8)
            return img

    def send_request(self, image_paths: List[str]):
        start = time.time()
        image_bytes = [self.preprocess_image(image_path) for image_path in image_paths]
        end = time.time()

        print(f"Execution time to read images: {end - start:.4f} seconds")

        start = time.time()
        max_length = max(len(img) for img in image_bytes)
        batch_data = np.zeros((len(image_bytes), max_length), dtype=np.uint8)
        for i, img in enumerate(image_bytes):
            batch_data[i, :len(img)] = img
        end = time.time()
        print(f"Execution time to pad images: {end - start:.4f} seconds")

        input_tensor = httpclient.InferInput("INPUT_0", [batch_data.shape[0], batch_data.shape[1]], "UINT8")
        input_tensor.set_data_from_numpy(batch_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput("probabilities"),
            httpclient.InferRequestedOutput("predicted_class"),
        ]

        start = time.time()
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_tensor],
            outputs=outputs,
        )
        end = time.time()
        print(f"Execution time of request: {end - start:.4f} seconds")

        probabilities = response.as_numpy("probabilities")
        predicted_classes = response.as_numpy("predicted_class")
        return probabilities, predicted_classes


# Main Code
if __name__ == "__main__":
    TRITON_SERVER_URL = "localhost:8000"
    MODEL_NAME = "vit"

    IMAGE_PATHS = ["dog.jpg"]

    triton_client = TritonClient(TRITON_SERVER_URL, MODEL_NAME)

    probabilities, predicted_classes = triton_client.send_request(IMAGE_PATHS)

    for i, image_path in enumerate(IMAGE_PATHS):
        print(f"Image: {image_path}")
        print(f"  Olasılıklar: {probabilities[i]}")
        print(f"  Tahmin Edilen Sınıf: {predicted_classes[i]}")
