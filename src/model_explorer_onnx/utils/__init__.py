from __future__ import annotations

from onnxscript import ir


def get_function_graph_name(identifier: ir.OperatorIdentifier) -> str:
    """Get the name of the function graph."""
    name = f"[function] {identifier[0]}::{identifier[1]}"
    if identifier[2]:
        name += f"::{identifier[2]}"
    return name
