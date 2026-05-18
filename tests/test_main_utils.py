from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
from onnx import TensorProto, helper

from model_explorer_onnx.main import (
    ONNXAdapter,
    Settings,
    _DEFAULT_OPSET_VERSION,
    _parse_namespace,
    can_display_tensor_json,
    create_graph,
    create_op_label,
    display_tensor_json,
    display_tensor_repr,
    format_shape,
    format_tensor_shape,
    format_type,
    get_constant_namespace,
    get_node_input_param_name,
    get_node_namespace,
    get_node_output_param_name,
)


class MainUtilsTest(unittest.TestCase):
    def test_display_helpers(self):
        self.assertEqual(display_tensor_repr(None), "Data not available")
        self.assertEqual(
            display_tensor_json(np.array([1, 2, 3]), Settings()), "[1,2,3]"
        )
        self.assertEqual(
            display_tensor_json(
                np.array([1, 2, 3, 4]), Settings(const_element_count_limit=2)
            ),
            "[1,2]",
        )
        self.assertEqual(
            display_tensor_json(
                np.array([1, 2, 3]), Settings(const_element_count_limit=-1)
            ),
            "[1,2,3]",
        )
        self.assertTrue(can_display_tensor_json(np.array([1]), Settings()))
        self.assertFalse(can_display_tensor_json(None, Settings()))

        class BadTensor:
            def numpy(self):
                raise RuntimeError("broken")

        self.assertEqual(display_tensor_json(BadTensor(), Settings()), "")

    def test_format_helpers(self):
        self.assertEqual(format_shape(None), "[?]")
        self.assertEqual(format_type(None), "?")
        self.assertEqual(
            format_tensor_shape(SimpleNamespace(dtype="FLOAT", shape="[1]")), "FLOAT[1]"
        )

    def test_namespace_helpers(self):
        self.assertEqual(_parse_namespace("/a/b//c/"), ["a", "b", "c"])
        self.assertEqual(
            get_node_namespace(
                SimpleNamespace(metadata_props={"namespace": "/x/y/"}, name="n")
            ),
            ["x", "y"],
        )
        self.assertEqual(
            get_node_namespace(SimpleNamespace(metadata_props={}, name="/m/n/node")),
            ["m", "n"],
        )
        self.assertEqual(
            get_node_namespace(SimpleNamespace(metadata_props={}, name="")), []
        )

    def test_schema_param_name_helpers(self):
        concat_schema = onnx.defs.get_schema("Concat", max_inclusive_version=18)
        self.assertEqual(get_node_input_param_name(concat_schema, 0), "inputs")
        self.assertEqual(get_node_input_param_name(concat_schema, 10), "inputs")
        self.assertIsNone(get_node_output_param_name(concat_schema, 1))

        split_schema = onnx.defs.get_schema("Split", max_inclusive_version=18)
        self.assertEqual(get_node_output_param_name(split_schema, 0), "outputs")
        self.assertEqual(get_node_output_param_name(split_schema, 10), "outputs")

    def test_create_op_label(self):
        self.assertEqual(create_op_label("", "Add"), "Add")
        self.assertEqual(create_op_label("ai.onnx", "Add"), "Add")
        self.assertEqual(create_op_label("custom.domain", "Add"), "custom.domain::Add")

    def test_get_constant_namespace_limits_to_shortest_user_namespace(self):
        class HashableNode:
            def __init__(self, namespace: str, name: str):
                self.metadata_props = {"namespace": namespace}
                self.name = name

        initializer = SimpleNamespace(
            uses=lambda: [
                (HashableNode("a/b", "n1"), 0),
                (HashableNode("a", "n2"), 1),
            ]
        )
        self.assertEqual(get_constant_namespace(initializer, "root"), "root/a")

    def test_create_graph_covers_initializers_and_unnamed_nodes(self):
        graph = helper.make_graph(
            nodes=[helper.make_node("Add", ["X", "W"], ["Y"])],
            name="main_graph",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, [1]),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
            initializer=[
                helper.make_tensor("W", TensorProto.FLOAT, [1], [1.0]),
                helper.make_tensor("UNUSED", TensorProto.FLOAT, [1], [3.0]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        ir_model = ir.serde.deserialize_model(model)

        rendered = create_graph(
            ir_model.graph,
            all_function_ids=set(),
            opset_version=18,
            settings=Settings(),
            attrs={},
        )
        assert rendered is not None

        by_id = {node.id: node for node in rendered.nodes}
        self.assertIn("<node_0>", by_id)
        self.assertEqual(by_id["[value] W"].label, "InitializedInput")
        self.assertEqual(by_id["[value] UNUSED"].label, "Initializer")

        unused_metadata = {
            item.key: item.value
            for item in by_id["[value] UNUSED"].outputsMetadata[0].attrs
        }
        self.assertEqual(unused_metadata["unused"], "True")

    def test_create_graph_with_missing_name_returns_none(self):
        self.assertIsNone(
            create_graph(
                SimpleNamespace(name=None),
                all_function_ids=set(),
                opset_version=18,
                settings=Settings(),
                attrs={},
            )
        )

    @mock.patch("model_explorer_onnx.main.create_graph")
    @mock.patch("model_explorer_onnx.main._passes.process_model")
    @mock.patch("model_explorer_onnx.main.ir.serde.deserialize_model")
    @mock.patch("model_explorer_onnx.main.onnx.shape_inference.infer_shapes")
    @mock.patch("model_explorer_onnx.main.onnx.load")
    def test_convert_fallbacks_and_subgraph_and_function_handling(
        self,
        mock_load,
        mock_infer_shapes,
        mock_deserialize,
        _mock_process_model,
        mock_create_graph,
    ):
        mock_load.return_value = object()
        mock_infer_shapes.side_effect = RuntimeError("shape inference failed")

        subgraph = SimpleNamespace(name=None)
        graph_attr = SimpleNamespace(
            type=ir.AttributeType.GRAPH, value=subgraph, name="body"
        )
        graph_node = SimpleNamespace(
            name="branch",
            attributes={"body": graph_attr},
            metadata_props={},
            doc_string="",
        )

        class FakeGraph(list):
            pass

        main_graph = FakeGraph([graph_node])
        main_graph.name = None
        main_graph.metadata_props = {}
        function = SimpleNamespace(attributes={}, metadata_props={}, doc_string="")
        fake_model = SimpleNamespace(
            graph=main_graph,
            functions={("custom", "Func", ""): function},
            opset_imports={},
            producer_name="",
            producer_version="",
            domain="",
            model_version=0,
            doc_string="",
            metadata_props={},
        )
        mock_deserialize.return_value = fake_model
        main_result = SimpleNamespace(id="main")
        function_result = SimpleNamespace(id="function")
        mock_create_graph.side_effect = [main_result, None, function_result]

        result = ONNXAdapter().convert("fake.onnx", {})

        self.assertEqual(fake_model.graph.name, "<main>")
        self.assertEqual(subgraph.name, "branch/body")
        self.assertEqual(result["graphs"], [main_result, function_result])
        self.assertEqual(mock_create_graph.call_count, 3)
        self.assertEqual(_DEFAULT_OPSET_VERSION, 18)

    def test_convert_with_real_model(self):
        graph = helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"], name="relu")],
            "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            onnx.save(model, path)
            converted = ONNXAdapter().convert(str(path), {})

        self.assertEqual(len(converted["graphs"]), 1)
        self.assertEqual(converted["graphs"][0].id, "g")


if __name__ == "__main__":
    unittest.main()
