from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import onnx_ir as ir

from model_explorer_onnx import _passes


class _FakeAttr:
    def __init__(self, type_, graph=None, graphs=None):
        self.type = type_
        self._graph = graph
        self._graphs = graphs or []

    def as_graph(self):
        return self._graph

    def as_graphs(self):
        return self._graphs


class _SafeDict(dict):
    def items(self):
        return list(super().items())


class _FakeGraph(list):
    def __init__(self, nodes=(), outputs=()):
        super().__init__(nodes)
        self.meta = {}
        self.outputs = list(outputs)
        self.removed = None

    def remove(self, nodes):
        self.removed = nodes

    __hash__ = object.__hash__


class _FakeOutput:
    def __init__(self, name):
        self.name = name
        self.type = "T"
        self.shape = "S"
        self.metadata_props = {}
        self.replaced_with = None

    def replace_all_uses_with(self, value, replace_graph_outputs=False):
        self.replaced_with = (value, replace_graph_outputs)


class _FakeNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.resize_calls = []
        self.replace_calls = []
        self.appended = None

    def resize_inputs(self, count):
        self.resize_calls.append(count)
        while len(self.inputs) < count:
            self.inputs.append(None)

    def replace_input_with(self, index, value):
        self.replace_calls.append((index, value))
        if len(self.inputs) <= index:
            self.inputs.extend([None] * (index + 1 - len(self.inputs)))
        self.inputs[index] = value

    def append(self, nodes):
        self.appended = nodes


class PassesTest(unittest.TestCase):
    def test_parse_namespace_and_get_node_namespace(self):
        self.assertEqual(_passes._parse_namespace("/a/b/"), ["a", "b"])
        self.assertEqual(
            _passes.get_node_namespace(SimpleNamespace(metadata_props={"namespace": "/x/y/"}, name="n")),
            ["x", "y"],
        )
        self.assertEqual(
            _passes.get_node_namespace(SimpleNamespace(metadata_props={}, name="/m/n/node")),
            ["m", "n"],
        )
        self.assertEqual(_passes.get_node_namespace(SimpleNamespace(metadata_props={}, name=None)), [])

    def test_assign_node_namespace_pass(self):
        n1 = SimpleNamespace(metadata_props={}, name="/a/node")
        n2 = SimpleNamespace(metadata_props={"namespace": "already"}, name="/b/node")
        n3 = SimpleNamespace(metadata_props={}, name="")
        model = SimpleNamespace(graph=SimpleNamespace(all_nodes=lambda: [n1, n2, n3]))
        result = _passes.AssignNodeNamespacePass()(model)
        self.assertTrue(result.modified)
        self.assertEqual(n1.metadata_props["namespace"], "a")
        self.assertEqual(n2.metadata_props["namespace"], "already")
        self.assertNotIn("namespace", n3.metadata_props)

    def test_implicit_use_analysis_pass_handles_graph_and_graphs_attributes(self):
        outer_graph = object()
        class _HashableValue:
            def __init__(self, graph):
                self.graph = graph

        captured = _HashableValue(outer_graph)

        subgraph_a = _FakeGraph(nodes=[SimpleNamespace(inputs=[captured], attributes={})])
        subgraph_b = _FakeGraph(nodes=[SimpleNamespace(inputs=[captured], attributes={})])
        top_node = SimpleNamespace(
            attributes={
                "g": _FakeAttr(ir.AttributeType.GRAPH, graph=subgraph_a),
                "gs": _FakeAttr(ir.AttributeType.GRAPHS, graphs=[subgraph_b]),
            }
        )
        model = SimpleNamespace(graph=[top_node])

        with mock.patch("model_explorer_onnx._passes.ir.Attr", _FakeAttr):
            result = _passes.ImplicitUseAnalysisPass()(model)
        self.assertFalse(result.modified)
        self.assertEqual(subgraph_a.meta["implicit_uses"], [captured])
        self.assertEqual(subgraph_b.meta["implicit_uses"], [captured])

    @mock.patch("model_explorer_onnx._passes.ir.node")
    def test_add_capture_node_pass_adds_capture_for_non_if_nodes(self, mock_ir_node):
        captured_output = SimpleNamespace(name="capture_out")
        capture_node = SimpleNamespace(outputs=[captured_output])
        mock_ir_node.return_value = capture_node

        subgraph = _FakeGraph()
        class _HashableValue:
            def __init__(self, name):
                self.name = name

        v = _HashableValue("v")
        subgraph.meta["implicit_uses"] = [v, v]
        graph_attr = _FakeAttr(ir.AttributeType.GRAPH, graph=subgraph)
        parent_graph = SimpleNamespace(insert_before=mock.Mock())

        non_if_node = _FakeNode(
            name="loop",
            op_type="Loop",
            domain="",
            attributes={"body": graph_attr},
            graph=parent_graph,
            inputs=[SimpleNamespace(name="in0")],
        )
        if_node = _FakeNode(
            name="ifnode",
            op_type="If",
            domain="",
            attributes={"body": graph_attr},
            graph=parent_graph,
            inputs=[SimpleNamespace(name="in0")],
        )
        model = SimpleNamespace(graph=SimpleNamespace(all_nodes=lambda: [if_node, non_if_node]))

        result = _passes.AddCaptureNodePass()(model)

        self.assertTrue(result.modified)
        parent_graph.insert_before.assert_called_once_with(non_if_node, capture_node)
        self.assertEqual(non_if_node.resize_calls, [2])
        self.assertEqual(non_if_node.replace_calls, [(1, captured_output)])
        mock_ir_node.assert_called_once()

    @mock.patch("model_explorer_onnx._passes.ir.node")
    def test_embed_if_pass_embeds_subgraph_and_creates_phi_node(self, mock_ir_node):
        phi_output = _FakeOutput("phi_out")
        phi_node = _FakeNode(outputs=[phi_output], inputs=[None], metadata_props={})
        mock_ir_node.return_value = phi_node

        sub_output = _FakeOutput("sub_out")
        sub_node = _FakeNode(
            name="body_node",
            metadata_props={},
            outputs=[sub_output],
            append=mock.Mock(),
        )
        graph_output = _FakeOutput("if_out")
        subgraph = _FakeGraph(nodes=[sub_node], outputs=[graph_output])
        if_attr = _FakeAttr(ir.AttributeType.GRAPH, graph=subgraph)

        if_output = _FakeOutput("main_out")
        if_node = _FakeNode(
            op_type="If",
            domain="",
            name="if0",
            metadata_props={},
            attributes=_SafeDict({"then_branch": if_attr}),
            outputs=[if_output],
            append=mock.Mock(),
        )
        model = SimpleNamespace(graph=SimpleNamespace(all_nodes=lambda: [if_node]))

        result = _passes.EmbedIfPass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(if_node.metadata_props["namespace"], "If <if0>")
        self.assertEqual(sub_node.name, "if0/then_branch/body_node")
        self.assertEqual(sub_output.name, "if0/then_branch/sub_out")
        self.assertEqual(graph_output.metadata_props["graph_output_index"], "0")
        self.assertEqual(list(if_node.attributes.keys()), [])
        if_node.append.assert_called_once()
        sub_node.append.assert_called_once_with(phi_node)
        self.assertEqual(if_output.replaced_with, (phi_output, True))
        self.assertEqual(if_output.name, "main_out_if_")
        self.assertEqual(phi_node.replace_calls, [(0, if_output)])

    @mock.patch("model_explorer_onnx._passes.common_passes.NameFixPass")
    @mock.patch("model_explorer_onnx._passes.ir.passes.PassManager")
    def test_process_model_runs_pass_manager_in_expected_order(
        self, mock_pass_manager_cls, mock_name_fix_cls
    ):
        model = SimpleNamespace()
        pass_manager_instance = mock.Mock()
        mock_pass_manager_cls.return_value = pass_manager_instance

        _passes.process_model(model)

        mock_pass_manager_cls.assert_called_once()
        called_passes = mock_pass_manager_cls.call_args.args[0]
        self.assertEqual(called_passes[0], mock_name_fix_cls.return_value)
        self.assertIsInstance(called_passes[1], _passes.AssignNodeNamespacePass)
        self.assertIsInstance(called_passes[2], _passes.ImplicitUseAnalysisPass)
        self.assertIsInstance(called_passes[3], _passes.AddCaptureNodePass)
        self.assertIsInstance(called_passes[4], _passes.EmbedIfPass)
        pass_manager_instance.assert_called_once_with(model)


if __name__ == "__main__":
    unittest.main()
