import pathlib

import onnx
from onnxscript import ir
import numpy as np


def main():
    input_ = ir.Input(
        "input", ir.Shape([42]), ir.TensorType(ir.DataType.FLOAT), "An input"
    )
    input_initialized = ir.Input(
        "input_initialized",
        ir.Shape([3]),
        ir.TensorType(ir.DataType.FLOAT),
        "An initialized input",
    )
    input_initializer = ir.Tensor(
        np.array([0.0, 1.0, 2.0], dtype=np.float32), name="input_initialized"
    )
    input_initialized.const_value = input_initializer
    normal_value = ir.Value(name="normal_initializer")
    normal_initializer = ir.Tensor(
        np.array([42], dtype=np.int64), name="normal_initializer"
    )
    normal_value.const_value = normal_initializer
    output = ir.Value(
        shape=ir.Shape([42]), type=ir.TensorType(ir.DataType.FLOAT), name="output"
    )
    output_initialized = ir.Value(
        shape=ir.Shape([3]),
        type=ir.TensorType(ir.DataType.FLOAT),
        name="output_initialized",
    )
    graph = ir.Graph(
        [input_, input_initialized],
        [output, output_initialized],
        nodes=[
            ir.Node(
                "",
                "Identity",
                [input_],
                outputs=[output],
                doc_string="An identity node 1",
            ),
            ir.Node(
                "",
                "Identity",
                [input_initialized],
                outputs=[output_initialized],
                doc_string="An identity node 2",
            ),
            ir.Node("", "Identity", [normal_value], doc_string="An identity node 3"),
        ],
        initializers=[input_initialized, normal_value],
        opset_imports={"": 20},
        name="initializer_node_tests",
    )
    model = ir.Model(graph, ir_version=8)
    model_proto = ir.serde.serialize_model(model)
    onnx.save(
        model_proto,
        pathlib.Path(__file__).parent.parent.parent
        / "testdata"
        / "initializer_node_tests.textproto",
    )


if __name__ == "__main__":
    main()
