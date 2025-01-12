import logging
import time

import tritonclient.http as httpclient
import numpy as np
import os

from utils.logger import setup_logger


def log_response_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.getLogger("TritonClient").info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds.")
        return result

    return wrapper


class TritonClient:
    def __init__(self, server_url: str, model_name: str):
        self.client = httpclient.InferenceServerClient(url=server_url)
        self.model_name = model_name
        self.logger = setup_logger()

    @log_response_time
    def send_request(self, base64_strings: list):
        self.logger.info("Sending a single inference request with all base64 strings...")

        filtered_strings = [s for s in base64_strings if s]
        if not filtered_strings:
            self.logger.error("No valid base64 strings to process.")
            return None

        try:
            input_tensor = httpclient.InferInput("INPUT_BASE64", [len(filtered_strings), 1], "BYTES")
            input_tensor.set_data_from_numpy(np.array(filtered_strings, dtype=np.object_).reshape(-1, 1))

            response = self.client.infer(
                model_name=self.model_name,
                inputs=[input_tensor],
                outputs=[
                    httpclient.InferRequestedOutput("probabilities"),
                    httpclient.InferRequestedOutput("predicted_class"),
                ],
            )

            results = [
                {
                    "probabilities": response.as_numpy("probabilities")[i],
                    "predicted_class": response.as_numpy("predicted_class")[i],
                }
                for i in range(len(filtered_strings))
            ]

            self.logger.info("Inference request completed successfully.")
            return results

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None


def main():
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

    TRITON_SERVER_URL = "localhost:8000"
    MODEL_NAME = "vit"
    BASE64_FOLDER = "base64"

    triton_client = TritonClient(TRITON_SERVER_URL, MODEL_NAME)

    file_list = ["cat.txt", "cat_2.txt", "dog.txt"]

    base64_strings = []
    for filename in file_list:
        file_path = os.path.join(BASE64_FOLDER, filename)
        try:
            with open(file_path, "r") as file:
                content = file.read().strip()
                if content:
                    base64_strings.append(content)
                    logging.info(f"Loaded base64 from {filename}")
                else:
                    logging.warning(f"Skipped empty file: {filename}")
        except Exception as e:
            logging.error(f"Failed to load {filename}: {e}")

    if base64_strings:
        responses = triton_client.send_request(base64_strings)

        if responses:
            logging.info("Inference Results:")
            for i, response in enumerate(responses):
                logging.info(f"Image {i + 1}:")
                logging.info(f"  Probabilities: {response['probabilities']}")
                logging.info(f"  Predicted Class: {response['predicted_class']}")
        else:
            logging.error("Failed to get inference results.")
    else:
        logging.warning("No valid base64 strings loaded from the specified files.")


if __name__ == "__main__":
    main()
