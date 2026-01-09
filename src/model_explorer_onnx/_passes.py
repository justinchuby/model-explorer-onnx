from __future__ import annotations

import logging
import onnx_ir as ir
import onnx_ir.passes.common as common_passes


logger = logging.getLogger(__name__)


def _parse_namespace(node_name: str) -> list[str]:
    """Parse the namespace from the node name if it is in the format of /namespace/node_name."""
    split = node_name.lstrip("/").rstrip("/").split("/")
    return [ns for ns in split if ns != ""]


def get_node_namespace(node: ir.Node) -> list[str]:
    """Get the namespace from the node."""
    if (metadata_namespace := node.metadata_props.get("namespace")) is not None:
        return _parse_namespace(metadata_namespace)
    if node.name:
        ns = _parse_namespace(node.name)
        if not ns:
            return []
        # Remove the last part of the node name to get the namespace
        return ns[:-1]
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
                node.metadata_props["namespace"] = "/".join(get_node_namespace(node))
                modified = True
        return ir.passes.PassResult(model, modified=modified)


class AddCaptureNodePass(ir.passes.InPlacePass):
    """Add a (Capture) node to nodes with subgraphs to visualize closed variables.

    For nodes with subgraphs (e.g., If, Loop, Scan), find all values that are captured
    from the outer graph (closed variables) and create a special (Capture) node that
    takes these values as inputs. The Capture node is added as an input to the parent node.

    This pass uses a two-phase approach:
    1. Analysis phase: Find all closed variables for each node with subgraphs
    2. Transform phase: Add Capture nodes based on the analysis results
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Phase 1: Analysis - collect all closed values for each node
        node_closed_values = self._analyze_closed_values(model)

        # Phase 2: Transform - add Capture nodes based on analysis
        modified = self._transform_add_capture_nodes(model, node_closed_values)

        return ir.passes.PassResult(model, modified=modified)

    def _analyze_closed_values(self, model: ir.Model) -> dict[ir.Node, list[ir.Value]]:
        """Analysis phase: Find all closed values for each node with subgraphs.

        Uses depth-first traversal to process deepest subgraphs first, collecting
        closed values bottom-up.

        Returns:
            A dictionary mapping nodes to their list of closed values.
        """
        node_closed_values = {}
        self._analyze_graph_depth_first(model.graph, node_closed_values)
        return node_closed_values

    def _analyze_graph_depth_first(
        self,
        graph: ir.Graph,
        node_closed_values: dict[ir.Node, list[ir.Value]]
    ) -> None:
        """Recursively analyze a graph in depth-first order to find closed values.

        Args:
            graph: The graph to analyze.
            node_closed_values: Dictionary to populate with nodes and their closed values.
        """
        for node in graph:
            # First, recursively process any subgraphs (depth-first)
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    nested_subgraph = attr.value
                    self._analyze_graph_depth_first(nested_subgraph, node_closed_values)

            # Then, collect closed values for this node if it has subgraphs
            all_closed_values = []
            seen_values = set()

            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    subgraph = attr.value
                    closed_values = self._find_closed_values_for_graph(subgraph)

                    # Add to the list, avoiding duplicates
                    for value in closed_values:
                        if value not in seen_values:
                            seen_values.add(value)
                            all_closed_values.append(value)

            if all_closed_values:
                node_closed_values[node] = all_closed_values

    def _find_closed_values_for_graph(self, subgraph: ir.Graph) -> list[ir.Value]:
        """Find closed values for a specific graph (non-recursive for direct children).

        Args:
            subgraph: The graph to find closed values for.

        Returns:
            List of closed values (values from outer scopes).
        """
        used_values = []

        for node in subgraph:
            # Process nested subgraphs
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    nested_subgraph = attr.value
                    nested_closed_values = self._find_closed_values_for_graph(nested_subgraph)

                    # Values closed in nested subgraph that don't belong to current subgraph
                    # are also closed at this level
                    for val in nested_closed_values:
                        if val.graph is not subgraph:
                            used_values.append(val)

            # Check direct inputs to this node
            for inp in node.inputs:
                if inp is not None and inp.graph is not subgraph:
                    # This is a closed value from outer scope
                    used_values.append(inp)

        # Remove duplicates while preserving order
        seen = set()
        closed_values = []
        for val in used_values:
            if val not in seen:
                seen.add(val)
                closed_values.append(val)

        return closed_values

    def _transform_add_capture_nodes(
        self, model: ir.Model, node_closed_values: dict[ir.Node, list[ir.Value]]
    ) -> bool:
        """Transform phase: Add Capture nodes for nodes with closed values.

        Args:
            model: The model to transform.
            node_closed_values: Dictionary mapping nodes to their closed values.

        Returns:
            True if the model was modified, False otherwise.
        """
        modified = False

        for node, all_closed_values in node_closed_values.items():
            # Create a Capture node
            capture_node = ir.node(
                "(Capture)",
                inputs=all_closed_values,
                num_outputs=len(all_closed_values),
                name=f"{node.name}_capture",
            )

            # Set output properties to match input properties
            for i, input_value in enumerate(all_closed_values):
                output = capture_node.outputs[i]
                output.name = f"{node.name}_captured_{input_value.name}"
                output.type = input_value.type
                output.shape = input_value.shape

            # Find the graph containing the node and insert the capture node before it
            graph = node.graph
            if graph is not None:
                graph.insert_before(node, capture_node)

                # Add the capture node's output as an input to the current node
                # Use resize_inputs to add a new input slot
                original_input_count = len(node.inputs)
                node.resize_inputs(original_input_count + 1)
                node.replace_input_with(
                    original_input_count, capture_node.outputs[0]
                )

                modified = True

        return modified


class EmbedIfPass(ir.passes.InPlacePass):
    """Convert the model to embed If directly within nodes."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if node.op_type != "If" or node.domain != "":
                continue
            modified = True
            # Handle single graph attributes only
            subgraphs: list[tuple[str, ir.Graph]] = []
            parent_namespace = get_node_namespace(node)
            parent_namespace.append(f"{node.op_type} <{node.name}>")
            node.metadata_props["namespace"] = "/".join(parent_namespace)
            for name, attr in node.attributes.items():
                if attr.type == ir.AttributeType.GRAPH:
                    subgraphs.append((name, attr.as_graph()))

            if not subgraphs:
                logger.warning("IF node %s has no subgraphs", node.name)
                continue

            outputs = []
            last_node = None
            for attr_name, subgraph in subgraphs:
                # Assign namespace to all nodes in the subgraph
                for sub_node in subgraph:
                    sub_namespace = get_node_namespace(sub_node)
                    sub_node.metadata_props["namespace"] = "/".join(
                        (*parent_namespace, attr_name, *sub_namespace)
                    )
                    sub_node.name = f"{node.name}/{attr_name}/{sub_node.name}"
                    for idx, output in enumerate(sub_node.outputs):
                        output.name = f"{node.name}/{attr_name}/{output.name}"
                for idx, g_output in enumerate(subgraph.outputs):
                    g_output.metadata_props["graph_output_index"] = str(idx)
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
                "(Phi)",
                inputs=(*([None] * len(node.outputs)), *[out for out in outputs]),
                num_outputs=len(node.outputs),
                name=f"{node.name}_phi",
                metadata_props=node.metadata_props.copy(),
            )
            last_node.append(phi_node)
            # Copy all information from the If node to the Phi node
            for i, output in enumerate(node.outputs):
                phi_output = phi_node.outputs[i]
                phi_output.name = output.name
                phi_output.type = output.type
                phi_output.shape = output.shape
                output.replace_all_uses_with(phi_output, replace_graph_outputs=True)
                output.name = f"{output.name}_if_"
            for i, output in enumerate(node.outputs):
                phi_node.replace_input_with(i, output)
        return ir.passes.PassResult(model, modified=modified)


def process_model(model: ir.Model) -> None:
    """Process the model with the defined passes."""
    passes = ir.passes.PassManager(
        [
            common_passes.NameFixPass(),
            AssignNodeNamespacePass(),
            AddCaptureNodePass(),
            EmbedIfPass(),
        ]
    )
    passes(model)
