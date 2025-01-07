import numpy as np
from transformers import LayoutLMv2Processor
from paddleocr import PaddleOCR
from PIL import Image
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        self.ocr = None
        self.processor = None

    def initialize(self, args):
        print("Preprocessing image...")
        self.processor = LayoutLMv2Processor.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", revision="no_ocr"
        )
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            ocr_version='PP-OCRv3'
        )

    def execute(self, requests):
        input_images = [
            pb_utils.get_input_tensor_by_name(request, "image").as_numpy()[0]
            for request in requests
        ]

        batch_results = [
            self._process_single_image(image) for image in input_images
        ]

        input_ids = np.concatenate([result["input_ids"] for result in batch_results], axis=0)
        bboxes = np.concatenate([result["bbox"] for result in batch_results], axis=0)
        processed_images = np.stack([result["processed_image"] for result in batch_results], axis=0)

        input_ids_tensor = pb_utils.Tensor("input_ids", input_ids)
        bbox_tensor = pb_utils.Tensor("bbox", bboxes)
        processed_image_tensor = pb_utils.Tensor("processed_image", processed_images)

        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[input_ids_tensor, bbox_tensor, processed_image_tensor]
            )
            for _ in requests
        ]

        return responses

    def _process_single_image(self, image):
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

        height, width = image.shape[:2]

        print(f"Image Dimensions: Height={height}, Width={width}")

        ocr_results = self.ocr.ocr(image, cls=False)
        words, boxes = [], []

        for line in ocr_results[0]:
            if len(line) == 2 and isinstance(line[0], list) and isinstance(line[1], tuple):
                bbox = line[0]
                word = line[1][0]

                x_min = min([point[0] for point in bbox])
                y_min = min([point[1] for point in bbox])
                x_max = max([point[0] for point in bbox])
                y_max = max([point[1] for point in bbox])

                words.append(word)
                boxes.append([
                    int(x_min * 1000 / width),
                    int(y_min * 1000 / height),
                    int(x_max * 1000 / width),
                    int(y_max * 1000 / height),
                ])
            else:
                print(f"Unexpected word_info format: {line}")

        encoded_inputs = self.processor(
            Image.fromarray(image).convert("RGB"),
            words,
            boxes=boxes,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        input_ids = encoded_inputs["input_ids"].astype(np.int64)
        bbox = encoded_inputs["bbox"].astype(np.int64)
        processed_image = encoded_inputs["image"].squeeze(0).astype(np.uint8)

        print(f"Input Ids Dimensions: {input_ids.shape}")
        print(f"Bbox Dimensions: {bbox.shape}")
        print(f"Processed Image Dimensions: {processed_image.shape}")

        return {
            "input_ids": input_ids,
            "bbox": bbox,
            "processed_image": processed_image,
        }

    def finalize(self):
        print("Preprocessing model is being finalized.")
