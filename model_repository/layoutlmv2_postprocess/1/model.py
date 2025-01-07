import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        self.id2label = None

    def initialize(self, args):
        id2label_path = "/model_repository/layoutlmv2_postprocess/1/resource/id2label.json"
        with open(id2label_path, "r") as f:
            self.id2label = json.load(f)

    def execute(self, requests):
        responses = []

        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            bboxes = pb_utils.get_input_tensor_by_name(request, "bbox").as_numpy()

            predictions = np.argmax(logits, axis=-1)

            labels = [self.id2label[str(pred)] for pred in predictions.flatten()]

            labels_tensor = pb_utils.Tensor(
                "labels",
                np.array(labels, dtype=object)
            )
            bboxes_tensor = pb_utils.Tensor("bbox_out", bboxes.reshape(-1, 4))

            responses.append(pb_utils.InferenceResponse(output_tensors=[labels_tensor, bboxes_tensor]))

        return responses

    def finalize(self):
        print("Postprocessing model is being finalized.")
