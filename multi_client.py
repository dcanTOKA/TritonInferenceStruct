import time
import tritonclient.http as httpclient
import numpy as np
import os
from multiprocessing import Pool


class TritonClient:
    def __init__(self, server_url: str, model_name: str):
        self.client = httpclient.InferenceServerClient(url=server_url)
        self.model_name = model_name

    @staticmethod
    def load_base64_files(folder_path: str) -> list:
        base64_strings = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    base64_strings.append(file.read().strip())
                print(f"Loaded base64 from {filename}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        return base64_strings

    def send_request(self, base64_strings: list, client_id: int):
        batch_data = np.array(base64_strings, dtype=np.object_).reshape(-1, 1)

        input_tensor = httpclient.InferInput("INPUT_BASE64", [batch_data.shape[0], batch_data.shape[1]], "BYTES")
        input_tensor.set_data_from_numpy(batch_data)

        outputs = [
            httpclient.InferRequestedOutput("probabilities"),
            httpclient.InferRequestedOutput("predicted_class"),
        ]

        try:
            start = time.time()
            response = self.client.infer(
                model_name=self.model_name,
                inputs=[input_tensor],
                outputs=outputs,
            )
            end = time.time()

            time_elapsed = end - start
            print(f"Client {client_id}: Time elapsed: {time_elapsed:.2f} seconds")

            return {
                "probabilities": response.as_numpy("probabilities"),
                "predicted_class": response.as_numpy("predicted_class"),
            }
        except Exception as e:
            print(f"Client {client_id} failed: {e}")
            return None


def simulate_request(client_id, server_url, model_name, base64_string):
    triton_client = TritonClient(server_url, model_name)
    return triton_client.send_request([base64_string], client_id)


def simulate_requests(server_url, model_name, base64_folder, num_clients=8):
    base64_strings = TritonClient.load_base64_files(base64_folder)
    if not base64_strings:
        print("No base64 files found in the folder.")
        return

    if len(base64_strings) < num_clients:
        print("Not enough base64 files for the number of clients. Replicating files...")
        base64_strings *= (num_clients // len(base64_strings)) + 1
        base64_strings = base64_strings[:num_clients]

    with Pool(processes=num_clients) as pool:
        client_args = [(i, server_url, model_name, base64_strings[i]) for i in range(num_clients)]
        results = pool.starmap(simulate_request, client_args)

        for client_id, response in enumerate(results):
            if response:
                print(f"Client {client_id} Inference Results:")
                for i, probs in enumerate(response["probabilities"]):
                    print(f"  Probabilities: {probs}")
                    print(f"  Predicted Class: {response['predicted_class'][i]}")


if __name__ == "__main__":
    TRITON_SERVER_URL = "localhost:8000"
    MODEL_NAME = "vit"
    BASE64_FOLDER = "base64"
    start = time.time()
    simulate_requests(TRITON_SERVER_URL, MODEL_NAME, BASE64_FOLDER)
    end = time.time()

    time_elapsed = end - start

    print(f"Time elapsed TOTAL: {time_elapsed:.2f} seconds")
