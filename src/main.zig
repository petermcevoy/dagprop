const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const tensor = @import("tensor.zig");
pub const TensorDevice = tensor.TensorDeviceCPU;

const dag = @import("dag.zig");
const Dag = dag.Dag;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tensor_dev = TensorDevice.init(allocator); // Hide tensor_dev from here and only have it within DAG?

    var graph = Dag.init(allocator, &tensor_dev);
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
