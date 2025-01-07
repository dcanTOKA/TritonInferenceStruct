import gradio as gr
import requests
import numpy as np

TRITON_SERVER_URL = "http://localhost:8000/v2/models/layoutlmv2_onnx/infer"


def query_triton(image):
    image_array = np.array(image, dtype=np.uint8)  # Triton için uint8 olmalı

    image_array = np.transpose(image_array, (2, 0, 1))  # NHWC -> CHW
    image_array = np.expand_dims(image_array, axis=0)  # (1, C, H, W)

    payload = {
        "inputs": [
            {
                "name": "image",
                "shape": image_array.shape,  # [1, 3, H, W]
                "datatype": "UINT8",
                "data": image_array.flatten().tolist()
            }
        ],
        "outputs": [
            {
                "name": "labels"
            },
            {
                "name": "bbox_out"
            }
        ]
    }

    print(image_array.shape)

    response = requests.post(TRITON_SERVER_URL, json=payload)
    if response.status_code == 200:
        output = response.json()
        bbox_out = output["outputs"][0]["data"]
        labels = output["outputs"][1]["data"]
        return labels, bbox_out
    else:
        raise ValueError(f"Error: {response.status_code}\n{response.text}")


def infer_layoutlmv2(image):
    try:
        labels, bbox_out = query_triton(image)
        reshaped_bboxes = np.array(bbox_out).reshape(-1, 4)
        result = "\n".join([
            f"Label: {label}, BBox: {list(bbox)}"
            for label, bbox in zip(labels, reshaped_bboxes)
        ])
        return result
    except Exception as e:
        return f"Error: {str(e)}"


interface = gr.Interface(
    fn=infer_layoutlmv2,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="LayoutLMv2 Model Repository",
    description="Load an image. Get back the labels and bbox"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
