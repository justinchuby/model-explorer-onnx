import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from model_explorer_onnx.main import ONNXAdapter


class OptionalInputsTest(unittest.TestCase):
    def test_convert_handles_omitted_optional_inputs(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Resize", ["X", "", "", "sizes"], ["Y"], mode="nearest"
                ),
            ],
            "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 16, 16])],
            [helper.make_tensor("sizes", TensorProto.INT64, [4], [1, 3, 16, 16])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "repro.onnx"
            onnx.save(model, model_path)

            converted = ONNXAdapter().convert(str(model_path), {})

        self.assertEqual(len(converted["graphs"]), 1)

        resize_node = next(node for node in converted["graphs"][0].nodes if node.label == "Resize")
        self.assertEqual(
            [
                next(attr.value for attr in metadata.attrs if attr.key == "__tensor_tag")
                for metadata in resize_node.inputsMetadata
            ],
            ["X", "None", "None", "sizes"],
        )


if __name__ == "__main__":
    unittest.main()
