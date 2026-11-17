import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest
from onnx import TensorProto, helper

from model_explorer_onnx.torch_utils import save_node_data_from_verification_info


@pytest.mark.parametrize("reverse", [False, True])
def test_overlay_uses_graph_id_and_worst_error_per_metric(tmp_path, reverse):
    def tensor(name, dtype):
        return helper.make_tensor_value_info(name, dtype, [1])

    graph = helper.make_graph(
        [helper.make_node("Dropout", ["x"], ["y", "mask"], name="drop")],
        "custom_graph",
        [tensor("x", TensorProto.FLOAT)],
        [tensor("y", TensorProto.FLOAT), tensor("mask", TensorProto.BOOL)],
    )
    model = ir.serde.deserialize_model(helper.make_model(graph))
    infos = [
        SimpleNamespace(name="y", max_abs_diff=0.9, max_rel_diff=0.02),
        SimpleNamespace(name="mask", max_abs_diff=0.001, max_rel_diff=0.5),
    ]
    save_node_data_from_verification_info(
        infos[::-1] if reverse else infos, model, str(tmp_path)
    )
    for metric in ("max_abs_diff", "max_rel_diff"):
        data = json.loads((tmp_path / f"model_{metric}.json").read_text())
        results = data["custom_graph"]["results"]
        assert results["drop"]["value"] == max(getattr(info, metric) for info in infos)
        for info in infos:
            assert results[f"[value] {info.name}"]["value"] == getattr(info, metric)
