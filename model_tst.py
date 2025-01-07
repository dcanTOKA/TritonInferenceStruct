import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from transformers import LayoutLMv2Processor
import os

ort_session = ort.InferenceSession("model_repository/layoutlmv2_onnx_model/1/model.onnx")

processor = LayoutLMv2Processor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", revision="no_ocr"
)

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=True,
    ocr_version='PP-OCRv3'
)


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label


labels = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
id2label = {v: k for v, k in enumerate(labels)}
label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}

image_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for image_file in os.listdir(image_dir):
    if image_file.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        ocr_result = ocr.ocr(image_path)
        words, bboxes = [], []

        for line in ocr_result[0]:
            if len(line) == 2 and isinstance(line[0], list) and isinstance(line[1], tuple):
                bbox = line[0]
                word = line[1][0]

                x_min = min([point[0] for point in bbox])
                y_min = min([point[1] for point in bbox])
                x_max = max([point[0] for point in bbox])
                y_max = max([point[1] for point in bbox])

                words.append(word)
                bboxes.append([
                    int(x_min * 1000 / width),
                    int(y_min * 1000 / height),
                    int(x_max * 1000 / width),
                    int(y_max * 1000 / height),
                ])
            else:
                print(f"Unexpected word_info format: {line}")

        encoded_inputs = processor(
            image, words, boxes=bboxes, padding="max_length", truncation=True, return_tensors="pt"
        )

        ort_inputs = {
            "input_ids": encoded_inputs['input_ids'].cpu().numpy(),
            "bbox": encoded_inputs['bbox'].cpu().numpy(),
            "image": encoded_inputs['image'].cpu().numpy()
        }

        ort_outputs = ort_session.run(None, ort_inputs)

        predictions = np.argmax(ort_outputs[0], axis=-1).squeeze().tolist()
        token_boxes = encoded_inputs.bbox.squeeze().tolist()

        true_predictions = [id2label[prediction] for prediction in predictions]
        true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for prediction, box in zip(true_predictions, true_boxes):
            predicted_label = iob_to_label(prediction).lower()
            draw.rectangle(box, outline=label2color.get(predicted_label, "red"), width=2)
            draw.text(
                (box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color.get(predicted_label, "red"),
                font=font
            )
        output_path = os.path.join(output_dir, image_file)
        image.save(output_path)
        print(f"Processed and saved: {output_path}")

print("All images processed.")
