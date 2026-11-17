import runpy
from dataclasses import asdict
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from model_explorer_onnx.main import ONNXAdapter


@pytest.fixture(params=["native", "web"])
def convert(request, tmp_path):
    def run(graph):
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx.checker.check_model(model)
        path = tmp_path / "model.onnx"
        onnx.save(model, path)
        if request.param == "native":
            return [asdict(g) for g in ONNXAdapter().convert(str(path), {})["graphs"]]
        source = Path(__file__).resolve().parents[1] / "web/src/python/convert.py"
        return runpy.run_path(str(source))["convert_onnx_file"](str(path), {})["graphs"]

    return run


def value(name, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(
        name, dtype, [1] if dtype == TensorProto.FLOAT else []
    )


def test_passthrough(convert):
    (graph,) = convert(helper.make_graph([], "main", [value("x")], [value("x")]))
    inp, out = graph["nodes"]
    assert inp["id"] != out["id"]
    assert out["incomingEdges"][0]["sourceNodeId"] == inp["id"]


def test_if_branch_constants(convert):
    def branch(number, with_identity):
        initializer = helper.make_tensor("weight", TensorProto.FLOAT, [1], [number])
        nodes = (
            [helper.make_node("Identity", ["weight"], ["out"])] if with_identity else []
        )
        return helper.make_graph(
            nodes, "branch", [], [value("out" if nodes else "weight")], [initializer]
        )

    cond = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="choice",
        then_branch=branch(1, True),
        else_branch=branch(2, False),
    )
    (graph,) = convert(
        helper.make_graph(
            [cond], "main", [value("cond", TensorProto.BOOL)], [value("y")]
        )
    )
    nodes = graph["nodes"]
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes)
    assert sum(n["label"] == "Initializer" for n in nodes) == 2
    assert all(e["sourceNodeId"] in ids for n in nodes for e in n["incomingEdges"])
    phi = next(n for n in nodes if n["label"] == "(Phi)")
    assert len(phi["incomingEdges"]) == 3


def test_nested_loop(convert):
    def body(prefix, nested=None):
        cond, data = prefix + "cond", prefix + "data"
        nodes = [helper.make_node("Identity", [cond], [cond + "Out"])]
        nodes.append(
            helper.make_node(
                "Loop", ["trip", cond, data], [data + "Out"], name="inner", body=nested
            )
            if nested
            else helper.make_node("Identity", [data], [data + "Out"])
        )
        return helper.make_graph(
            nodes,
            prefix,
            [
                value(prefix + "index", TensorProto.INT64),
                value(cond, TensorProto.BOOL),
                value(data),
            ],
            [value(cond + "Out", TensorProto.BOOL), value(data + "Out")],
        )

    loop = helper.make_node(
        "Loop", ["trip", "cond", "x"], ["y"], name="outer", body=body("a", body("b"))
    )
    graphs = convert(
        helper.make_graph(
            [loop],
            "main",
            [
                value("trip", TensorProto.INT64),
                value("cond", TensorProto.BOOL),
                value("x"),
            ],
            [value("y")],
        )
    )
    assert len(graphs) == 3
    refs = {
        ref
        for graph in graphs
        for node in graph["nodes"]
        for ref in node["subgraphIds"]
    }
    assert refs == {graph["id"] for graph in graphs if graph["id"] != "main"}
