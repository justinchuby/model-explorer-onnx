"""
Example of creating an ONNX model with a Loop that references values from the outer graph.
This demonstrates how to use onnx_ir to build models with subgraphs.
"""

import onnx
from onnx import helper, TensorProto


def create_loop_model_with_outer_references():
    """
    Creates a Loop node that references values from the outer graph.

    The model computes:
    - Input: x (scalar), max_iterations (scalar), initial_value (scalar)
    - Loop body: accumulated_value = accumulated_value + x (x is from outer graph)
    - Output: final accumulated value after max_iterations
    """

    # Define the loop body subgraph
    # Loop iteration variables:
    # - iteration_num: current iteration number (i)
    # - cond: loop condition (boolean)
    # - accumulated: the accumulated sum value (state variable)

    # Subgraph inputs: [iteration_num, cond, accumulated]
    iter_input = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    cond_input = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    accum_input = helper.make_tensor_value_info("accum_in", TensorProto.FLOAT, [])

    # 'x_outer' is referenced from the outer graph (implicit input)
    # Inside the loop body, we add x_outer to accumulated value
    add_node = helper.make_node(
        "Add",
        inputs=["accum_in", "x_outer"],  # x_outer comes from outer graph
        outputs=["accum_out"],
    )

    # Output the condition (always true for fixed iterations)
    cond_output = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])

    # Output the updated accumulated value
    accum_output = helper.make_tensor_value_info("accum_out", TensorProto.FLOAT, [])

    # Identity node to pass condition through
    identity_node = helper.make_node(
        "Identity", inputs=["cond_in"], outputs=["cond_out"]
    )

    # Create the loop body subgraph
    loop_body = helper.make_graph(
        nodes=[add_node, identity_node],
        name="loop_body",
        inputs=[iter_input, cond_input, accum_input],
        outputs=[cond_output, accum_output],
    )

    # Create the main graph
    # Inputs to the main graph
    x_input = helper.make_tensor_value_info("x_outer", TensorProto.FLOAT, [])
    max_iter_input = helper.make_tensor_value_info("max_iter", TensorProto.INT64, [])
    initial_value_input = helper.make_tensor_value_info(
        "initial", TensorProto.FLOAT, []
    )

    # Create initial condition (true)
    true_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["initial_cond"],
        value=helper.make_tensor("const_true", TensorProto.BOOL, [], [True]),
    )

    # Create the Loop node
    # Loop inputs: [max_iterations, initial_condition, initial_state_variables...]
    # The loop body can access 'x_outer' from the outer graph implicitly
    loop_node = helper.make_node(
        "Loop",
        inputs=["max_iter", "initial_cond", "initial"],
        outputs=["final_result"],
        body=loop_body,
    )

    # Output
    output = helper.make_tensor_value_info("final_result", TensorProto.FLOAT, [])

    # Create the main graph
    main_graph = helper.make_graph(
        nodes=[true_constant, loop_node],
        name="loop_with_outer_reference",
        inputs=[x_input, max_iter_input, initial_value_input],
        outputs=[output],
    )

    # Create the model
    model = helper.make_model(main_graph, producer_name="onnx-loop-example")
    model.opset_import[0].version = 13  # Loop operator version

    return model


def create_loop_with_multiple_outer_refs():
    """
    Creates a more complex Loop that references multiple values from outer graph.

    Loop computes: result = (accum + x) * y where x and y are from outer graph
    """

    # Subgraph inputs
    iter_input = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    cond_input = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    accum_input = helper.make_tensor_value_info("accum_in", TensorProto.FLOAT, [1])

    # Add x_outer (from outer graph) to accumulated value
    add_node = helper.make_node(
        "Add", inputs=["accum_in", "x_outer"], outputs=["add_result"]
    )

    # Multiply by y_outer (also from outer graph)
    mul_node = helper.make_node(
        "Mul", inputs=["add_result", "y_outer"], outputs=["accum_out"]
    )

    # Pass condition through
    identity_node = helper.make_node(
        "Identity", inputs=["cond_in"], outputs=["cond_out"]
    )

    # Outputs
    cond_output = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    accum_output = helper.make_tensor_value_info("accum_out", TensorProto.FLOAT, [1])

    # Create loop body
    loop_body = helper.make_graph(
        nodes=[add_node, mul_node, identity_node],
        name="loop_body_multi_ref",
        inputs=[iter_input, cond_input, accum_input],
        outputs=[cond_output, accum_output],
    )

    # Main graph inputs
    x_input = helper.make_tensor_value_info("x_outer", TensorProto.FLOAT, [1])
    y_input = helper.make_tensor_value_info("y_outer", TensorProto.FLOAT, [1])
    max_iter_input = helper.make_tensor_value_info("max_iter", TensorProto.INT64, [])
    initial_input = helper.make_tensor_value_info("initial", TensorProto.FLOAT, [1])

    # Create initial condition
    true_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["initial_cond"],
        value=helper.make_tensor("const_true", TensorProto.BOOL, [], [True]),
    )

    # Create Loop node
    loop_node = helper.make_node(
        "Loop",
        inputs=["max_iter", "initial_cond", "initial"],
        outputs=["final_result"],
        body=loop_body,
    )

    # Output
    output = helper.make_tensor_value_info("final_result", TensorProto.FLOAT, [1])

    # Create main graph
    main_graph = helper.make_graph(
        nodes=[true_constant, loop_node],
        name="loop_with_multiple_outer_refs",
        inputs=[x_input, y_input, max_iter_input, initial_input],
        outputs=[output],
    )

    # Create model
    model = helper.make_model(main_graph, producer_name="onnx-loop-example")
    model.opset_import[0].version = 13

    return model


if __name__ == "__main__":
    # Create and save the first example
    model1 = create_loop_model_with_outer_references()
    onnx.checker.check_model(model1)
    onnx.save(model1, "loop_with_outer_ref.textproto")
    print("Created loop_with_outer_ref.textproto")

    # Create and save the second example
    model2 = create_loop_with_multiple_outer_refs()
    onnx.checker.check_model(model2)
    onnx.save(model2, "loop_with_multiple_refs.textproto")
    print("Created loop_with_multiple_refs.textproto")

    print("\nModel 1 structure:")
    print(model1)
    print("\nModel 2 structure:")
    print(model2)
