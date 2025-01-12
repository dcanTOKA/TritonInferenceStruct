import triton_python_backend_utils as pb_utils
import numpy as np
from PIL import Image
import base64
import io
from concurrent.futures import ThreadPoolExecutor


class TritonPythonModel:
    def __init__(self):
        self.executor = None

    def initialize(self, args):
        self.executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def _process_base64_image(base64_string):
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_bytes))
        original_size = img.size

        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return img_array, original_size

    def execute(self, requests):
        """
        Here is the workflow of execution:
            1- It is imported to note that we must be sure whether the requests will be processed in parallel.
            2- Convert base64 encoded images to padded numpy arrays. The reason why we do that is to make sure the shapes of the images in batches are same.
               Padded part is going to be removed in DALI via slicing method.
               In the end of this step we will get
               -> padded image   (batch_size x 1)
               -> original size  (batch_size x 2)
            3-
        :param requests:
        :return: list of InferenceResponse
        """
        responses = []

        for request in requests:
            base64_inputs = pb_utils.get_input_tensor_by_name(
                request, "INPUT_BASE64"
            ).as_numpy()

            base64_inputs = base64_inputs.flatten()

            base64_strings = [
                b.decode("utf-8") if isinstance(b, bytes) else str(b, "utf-8")
                for b in base64_inputs
            ]

            processed_results = list(
                self.executor.map(self._process_base64_image, base64_strings)
            )

            decoded_images = [res[0] for res in processed_results]
            original_sizes = [res[1] for res in processed_results]

            if len(decoded_images) > 0:
                max_length = max(len(img_arr) for img_arr in decoded_images)
            else:
                max_length = 0

            padded_images = np.zeros((len(decoded_images), max_length), dtype=np.uint8)
            for i, img_array in enumerate(decoded_images):
                padded_images[i, :len(img_array)] = img_array

            original_sizes_np = np.array(original_sizes, dtype=np.int32)

            output_tensor_0 = pb_utils.Tensor("OUTPUT_BYTES", padded_images)
            output_tensor_1 = pb_utils.Tensor("OUTPUT_SIZES", original_sizes_np)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor_0, output_tensor_1]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        self.executor.shutdown()
