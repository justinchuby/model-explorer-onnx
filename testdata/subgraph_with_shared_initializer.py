"""Create a model with a subgraph that uses an initializer from the main graph.

This model demonstrates:
1. An initializer in the main graph
2. A node in the main graph that uses this initializer
3. A subgraph (in an If node) that uses the same initializer
4. A node in the subgraph that uses the output of the node in the main graph
"""

import numpy as np
import onnx
import onnx_ir as ir

# Create a shared initializer tensor
shared_initializer_tensor = ir.tensor(
    np.array([2.0, 3.0], dtype=np.float32), name="shared_weight"
)

# Create the initializer value for the main graph
shared_initializer = ir.Value(
    name="shared_weight",
    shape=shared_initializer_tensor.shape,
    type=ir.TensorType(ir.DataType.FLOAT),
    const_value=shared_initializer_tensor,
)

# Create input value for the main graph
input_value = ir.Value(
    name="input", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((2,))
)

# Create a condition value for the If node
condition = ir.Value(
    name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(())
)

# Node in main graph that uses the shared initializer
# This multiplies the input by the shared weight
mul_node_main = ir.node("Mul", inputs=[input_value, shared_initializer], num_outputs=1)
mul_node_main.outputs[0].name = "main_output"
mul_node_main.outputs[0].dtype = ir.DataType.FLOAT
mul_node_main.outputs[0].shape = ir.Shape((2,))

# Create the then_branch subgraph
# This subgraph uses:
# 1. The shared initializer from the main graph
# 2. The output from mul_node_main
then_add_node = ir.node(
    "Add",
    inputs=[mul_node_main.outputs[0], shared_initializer],
    num_outputs=1,
)
then_add_node.outputs[0].name = "then_result"
then_add_node.outputs[0].dtype = ir.DataType.FLOAT
then_add_node.outputs[0].shape = ir.Shape((2,))

then_graph = ir.Graph(
    inputs=[],  # Empty inputs as it uses values from parent scope
    outputs=[then_add_node.outputs[0]],
    nodes=[then_add_node],
    name="then_branch",
)

# Create a simple else_branch subgraph
else_identity_node = ir.node(
    "Identity", inputs=[mul_node_main.outputs[0]], num_outputs=1
)
else_identity_node.outputs[0].name = "else_result"
else_identity_node.outputs[0].dtype = ir.DataType.FLOAT
else_identity_node.outputs[0].shape = ir.Shape((2,))

else_graph = ir.Graph(
    inputs=[],
    outputs=[else_identity_node.outputs[0]],
    nodes=[else_identity_node],
    name="else_branch",
)

# Create the If node with the subgraphs
if_node = ir.node(
    "If",
    inputs=[condition],
    attributes={"then_branch": then_graph, "else_branch": else_graph},
    num_outputs=1,
)
if_node.outputs[0].name = "output"
if_node.outputs[0].dtype = ir.DataType.FLOAT
if_node.outputs[0].shape = ir.Shape((2,))

# Create the main graph
main_graph = ir.Graph(
    inputs=[input_value, condition],
    outputs=[if_node.outputs[0]],
    nodes=[mul_node_main, if_node],
    initializers=[shared_initializer],
    opset_imports={"": 20},
    name="main_graph",
)

# Create and save the model
model = ir.Model(graph=main_graph, ir_version=10)

# Print the model
print(model)
print("\n" + "=" * 80)
print("Model structure:")
print("=" * 80)
print(f"Main graph nodes: {len(main_graph)}")
print(f"Main graph initializers: {list(main_graph.initializers.keys())}")
print(f"\nThen branch nodes: {len(then_graph)}")
print(
    f"Then branch uses shared_initializer: {shared_initializer in then_add_node.inputs}"
)
print(
    f"Then branch uses main_output: {mul_node_main.outputs[0] in then_add_node.inputs}"
)

# Save to file
proto = ir.to_proto(model)
onnx.checker.check_model(proto)
onnx.save(proto, "subgraph_with_shared_initializer.textproto")
