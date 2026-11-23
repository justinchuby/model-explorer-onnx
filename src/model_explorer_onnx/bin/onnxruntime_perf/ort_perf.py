"""Convert the performance trace from ORT json file to node data."""

import json
import os
from model_explorer import node_data_builder as ndb
from typing import Any
import argparse


def _get_node_runtime_micro_secs(perf_traces: list[dict[str, Any]]) -> dict[str, int]:
    """Get the runtime of each node from the performance traces."""
    node_runtimes: dict[str, int] = {}
    for trace in perf_traces:
        if trace.get("cat") != "Node":
            continue
        if "name" not in trace:
            continue
        name = trace["name"]
        if not name.endswith("_kernel_time"):
            continue
        node_name = name.split("_kernel_time")[0]
        node_runtimes[node_name] = trace["dur"]
    return node_runtimes


def _convert_node_data_for_model_explorer(
    node_runtimes: dict[str, int],
) -> ndb.ModelNodeData:
    # https://github.com/google-ai-edge/model-explorer/wiki/4.-API-Guide#create-custom-node-data
    # This API is unstable and may change in the future.

    main_graph_results: dict[str, ndb.NodeDataResult] = {
        key: ndb.NodeDataResult(value=value) for key, value in node_runtimes.items()
    }

    gradient: list[ndb.GradientItem] = [
        ndb.GradientItem(stop=0, bgColor="green"),
        ndb.GradientItem(stop=1, bgColor="red"),
    ]

    # # Construct the data for the main graph.
    main_graph_data = ndb.GraphNodeData(results=main_graph_results, gradient=gradient)

    # Construct the data for the model.
    # "main_graph" is defined in _core.py
    model_data = ndb.ModelNodeData(graphsData={"main_graph": main_graph_data})
    return model_data


def main(args) -> None:
    """Main function."""
    perf_trace_path = args.perf_trace_path
    with open(perf_trace_path, encoding="utf-8") as f:
        traces = json.load(f)
    node_runtimes = _get_node_runtime_micro_secs(traces)
    model_data = _convert_node_data_for_model_explorer(node_runtimes)
    perf_trace_dir = os.path.dirname(perf_trace_path)
    perf_trace_name = os.path.basename(perf_trace_path)
    model_data_path = os.path.join(perf_trace_dir, f"{perf_trace_name}.node_data.json")
    model_data.save_to_file(model_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("perf_trace_path", type=str)
    args = parser.parse_args()
    main(args)
