from __future__ import annotations

import onnx_ir as ir
import onnx_ir.passes.common as common_passes


class AssignUniqueGraphNamesPass(ir.passes.InPlacePass):
    """Assign unique names to graphs and nodes in the model."""

    def _assign_unique_graph_name(
        self, name: str | None, existing_names: set[str]
    ) -> str:
        if not name:
            base_name = "graph"
        else:
            base_name = name

        unique_name = base_name
        index = 1
        while unique_name in existing_names:
            unique_name = f"{base_name}_{index}"
            index += 1
        existing_names.add(unique_name)
        return unique_name

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        all_graph_names = set()
        modified = False
        for graph in model.graphs():
            if not graph.name or graph.name in all_graph_names:
                graph.name = self._assign_unique_graph_name(graph.name, all_graph_names)
                modified = True
        return ir.passes.PassResult(model, modified=modified)


def _parse_namespace(node_name: str) -> list[str]:
    """Parse the namespace from the node name if it is in the format of /namespace/node_name."""
    split = node_name.lstrip("/").rstrip("/").split("/")
    return [ns or "<anonymous>" for ns in split]


def get_node_namespace(node: ir.Node) -> list[str]:
    """Get the namespace from the node."""
    if (metadata_namespace := node.metadata_props.get("namespace")) is not None:
        return _parse_namespace(metadata_namespace)
    return []


class AssignNodeNamespacePass(ir.passes.InPlacePass):
    """Assign namespace to nodes based on their containing graph."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if "namespace" in node.metadata_props:
                continue
            if node.name:
                # Remove the last part of the node name to get the namespace
                node.metadata_props["namespace"] = "/".join(
                    _parse_namespace(node.name)[:-1]
                )
                modified = True
        return ir.passes.PassResult(model, modified=modified)


class EmbedIfPass(ir.passes.InPlacePass):
    """Convert the model to embed If directly within nodes."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if node.op_type != "If":
                continue
            modified = True
            # Handle only single graph attributes only
            subgraphs: list[tuple[str, ir.Graph]] = []
            parent_name_space = get_node_namespace(node)
            parent_name_space.append(f"{node.op_type}<{node.name}>")
            node.metadata_props["namespace"] = "/".join(parent_name_space)
            for name, attr in node.attributes.items():
                if attr.type == ir.AttributeType.GRAPH:
                    subgraphs.append((name, attr.as_graph()))

            outputs = []
            last_node = None
            for attr_name, subgraph in subgraphs:
                # Assign namespace to all nodes in the subgraph
                for sub_node in subgraph.all_nodes():
                    sub_namespace = get_node_namespace(sub_node)
                    sub_node.metadata_props["namespace"] = "/".join(
                        (*parent_name_space, attr_name, *sub_namespace)
                    )
                # Remove the attribute from the node
                node.attributes.pop(attr_name)
                outputs.extend(subgraph.outputs)
                # Clear the subgraph and move the nodes to the parent graph
                subgraph.outputs.clear()
                sub_nodes = tuple(subgraph)
                subgraph.remove(sub_nodes)
                node.append(sub_nodes)
                last_node = sub_nodes[-1]
            assert last_node is not None
            # Create a phi node to merge outputs
            phi_node = ir.node(
                "Phi",
                inputs=[out for out in outputs],
                num_outputs=len(node.outputs),
            )
            last_node.append(phi_node)
            # Copy all information from the If node to the Phi node
            for i, output in enumerate(node.outputs):
                phi_output = phi_node.outputs[i]
                phi_output.name = output.name
                phi_output.type = output.type
                phi_output.shape = output.shape
                output.replace_all_uses_with(phi_output)
                output.name = f"{output.name}_if_"
        return ir.passes.PassResult(model, modified=modified)


def process_model(model: ir.Model) -> None:
    """Process the model with the defined passes."""
    passes = ir.passes.PassManager([
        common_passes.NameFixPass(),
        AssignUniqueGraphNamesPass(),
        AssignNodeNamespacePass(),
        EmbedIfPass(),
    ])
    passes(model)
