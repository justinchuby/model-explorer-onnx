import math
import pathlib

import onnx
from onnxscript import ir


def main():
    graph = ir.Graph(
        [],
        [],
        nodes=[
            c1 := ir.Node(
                "", "Constant", [], attributes=[ir.AttrInt64("value_int", 42)]
            ),
            ir.Node("", "Identity", c1.outputs),
            c2 := ir.Node(
                "", "Constant", [], attributes=[ir.AttrInt64s("value_ints", [0, 42])]
            ),
            ir.Node("", "Identity", c2.outputs),
            c3 := ir.Node(
                "", "Constant", [], attributes=[ir.AttrFloat32("value_float", 42.0)]
            ),
            ir.Node("", "Identity", c3.outputs),
            c4 := ir.Node(
                "",
                "Constant",
                [],
                attributes=[ir.AttrFloat32s("value_floats", [0.0, math.pi])],
            ),
            ir.Node("", "Identity", c4.outputs),
            c5 := ir.Node(
                "", "Constant", [], attributes=[ir.AttrString("value_string", "hello")]
            ),
            ir.Node("", "Identity", c5.outputs),
            c6 := ir.Node(
                "",
                "Constant",
                [],
                attributes=[ir.AttrStrings("value_strings", ["hello", "world"])],
            ),
            ir.Node("", "Identity", c6.outputs),
        ],
        opset_imports={"": 20},
        name="constant_node_tests",
    )
    model = ir.Model(graph, ir_version=8)
    model_proto = ir.serde.serialize_model(model)
    onnx.save(
        model_proto,
        pathlib.Path(__file__).parent.parent.parent
        / "testdata"
        / "constant_node_tests.textproto",
    )


if __name__ == "__main__":
    main()
