import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        print("Postprocessing model initialized.")

    def execute(self, requests):
        responses = []

        for request in requests:
            logits_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            logits = logits_tensor.as_numpy()

            probabilities = self.softmax(logits)

            predicted_class = np.argmax(probabilities, axis=1).astype(np.int64)

            probabilities_tensor = pb_utils.Tensor("probabilities", probabilities.astype(np.float32))
            predicted_class_tensor = pb_utils.Tensor("predicted_class", predicted_class)

            responses.append(pb_utils.InferenceResponse(output_tensors=[probabilities_tensor, predicted_class_tensor]))

        return responses

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def finalize(self):
        print("Postprocessing model finalized.")
