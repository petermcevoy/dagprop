const std = @import("std");
const Allocator = std.mem.Allocator;

const dagprop = @import("dagprop");
const tensor = dagprop.tensor;
pub const TensorDevice = tensor.TensorDeviceCPU;

const Dag = dagprop.Dag;
const Mlp = dagprop.Mlp;

pub fn main() !void {

    // Input layer 3
    // Hidden layer 3 + bias
    // output layer 1
    //
    // Inputs  = [[0,0], [0,1], [1,0], [1,1]]
    // Outputs = [[0], [1], [1], [0]]
    //
    // MLP, dense

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tensor_dev = TensorDevice.init(allocator); // Hide tensor_dev from here and only have it within DAG?

    var graph = Dag.init(allocator, &tensor_dev);

    var mlp = Mlp.init(allocator, &graph);

    // Add hidden layer with 2 neurons and two inputs
    mlp.add(.Dense, 2, 2);

    // Add output layer
    mlp.add(.Dense, 1);

    var a = try graph.constant(try tensor_dev.createTensorScalar(1));
    var b = try graph.constant(try tensor_dev.createTensorScalar(2));
    var sum = try graph.add(a, b);
    var c = try graph.constant(try tensor_dev.createTensorScalar(6));
    var sub = try graph.mul(sum, c);
    try graph.resolveNode(sub, .Forward);
    const value = tensor_dev.tensorValues(graph.nodes.items[sub].value.?)[0];
    std.debug.print("Value of sub node is: {}\n", .{value});
    try std.testing.expectEqual(value, 18.0);

    try graph.resolveNode(sub, .Backward);
    const grad = tensor_dev.tensorValues(graph.nodes.items[a].grad)[0];
    std.debug.print("Backpropagated gradient of intial constant: {}\n", .{grad});
    try std.testing.expectEqual(grad, 6.0);
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
