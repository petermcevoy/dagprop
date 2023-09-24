const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const tensor = @import("tensor.zig");
const Tensor = tensor.Tensor;
const TensorDescriptor = tensor.TensorDescriptor;
pub const TensorDevice = tensor.TensorDeviceCPU;

const dag = @import("dag.zig");
const DAG = dag.DAG;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var tensor_dev = TensorDevice.init(allocator); // Hide tensor_dev from here and only have it within DAG?

    var graph = DAG.init(allocator, &tensor_dev);
    var a = try graph.constant(try tensor_dev.createTensorScalar(1));
    var b = try graph.constant(try tensor_dev.createTensorScalar(2));
    var sum = try graph.add(a, b);
    var c = try graph.constant(try tensor_dev.createTensorScalar(6));
    var sub = try graph.sub(sum, c);
    try graph.resolveNode(sub, .Forward);
    const value = tensor_dev.tensorValues(graph.nodes.items[sub].value.?)[0];
    std.debug.print("Value of sub node is: {}\n", .{value});
    try std.testing.expectEqual(value, -3.0);

    // TODOs:
    // - Get inspiration from ECS videos on how to get the handles..

    // Goal 1 recreate micrograd
    // Dag needs to support backprop
    // Neuron takes # of inputs that come to a neuron
    // When, created a random weight is created for each input [0, 1]
    // and a bias for the neuron [0, 1].
    // Call neuron with input x:
    //  act = w.dot(x) + b
    //  return act.tanh()
    // Next, a layer of neurons is defined (MLP)
    // The concepts above should define these operations in terms of the DAG primitives,
    // so that we can back propagate later.

    // Goal 2 is to be able to train and run MNIST dataset
    // Ops needed:
    // self.conv1 = nn.Conv2d(1, 32, 3, 1)
    // self.conv2 = nn.Conv2d(32, 64, 3, 1)
    // self.dropout1 = nn.Dropout(0.25)
    // self.dropout2 = nn.Dropout(0.5)
    // self.fc1 = nn.Linear(9216, 128)
    // self.fc2 = nn.Linear(128, 10)
    //
    // x = self.conv1(x)
    // x = F.relu(x)
    // x = self.conv2(x)
    // x = F.relu(x)
    // x = F.max_pool2d(x, 2)
    // x = self.dropout1(x)
    // x = torch.flatten(x, 1)
    // x = self.fc1(x)
    // x = F.relu(x)
    // x = self.dropout2(x)
    // x = self.fc2(x)
    // output = F.log_softmax(x, dim=1)
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
