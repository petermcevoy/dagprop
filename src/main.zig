const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

fn productOfElements(elements: []const usize) usize {
    var product: usize = 1;
    for (elements) |element| {
        product *= element;
    }
    return product;
}

const TensorDeviceCPU = struct {
    const Self = @This();

    const TensorCPU = struct {
        elements: []f32,
    };

    allocator: Allocator,
    backing_tensors: ArrayList(TensorCPU),

    fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator, .backing_tensors = ArrayList(TensorCPU).init(allocator) };
    }

    fn deinit(self: *Self) void {
        for (self.backing_tensors.items) |tensor| {
            self.allocator.free(tensor.elements);
        }
        self.backing_tensors.deinit();
    }

    fn createTensor(self: *Self, desc: TensorDescriptor) !Tensor {
        var num_elements: usize = 1;
        for (desc.dimensions_sizes) |dim| {
            assert(dim > 0); // Tensor dimension sizes must be > 0
            num_elements *= dim;
        }

        std.debug.print("Allocating TensorCPU with {} elements.\n", .{num_elements});
        var elements = try self.allocator.alloc(f32, num_elements);
        std.mem.set(f32, elements, 0.0);
        var new_tensor_handle = self.backing_tensors.items.len;
        try self.backing_tensors.append(TensorCPU{ .elements = elements });
        return Tensor{ .handle = new_tensor_handle, .descriptor = desc };
    }

    fn setTensorData(self: *Self, tensor: Tensor, data: []const f32) void {
        var backing_tensor = self.backing_tensors.items[tensor.handle];
        assert(productOfElements(tensor.descriptor.dimensions_sizes) == data.len);
        std.mem.copy(f32, backing_tensor.elements, data);
    }

    fn tensorOpAddition(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] += backing_tensor_b.elements[i];
        }
    }

    fn tensorValues(self: *Self, tensor_a: Tensor) []f32 {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        return backing_tensor_a.elements;
    }
};

// NOTE: I should treat this as a resource handle.
// The device creates the resource.
// Tensors can be stored on different devices or in memory
const TensorDescriptor = struct {
    dimensions_sizes: []const usize,
    // data_type: ,
};
const Tensor = struct {
    // type that says if it's cpu or not
    descriptor: TensorDescriptor,
    handle: usize,
};

const DAG = struct {
    const Self = @This();
    nodes: ArrayList(Node),
    edges: ArrayList(DirectedEdge),
    allocator: Allocator,

    pub fn init(allocator: Allocator) @This() {
        var ret = Self{ .allocator = allocator, .nodes = std.ArrayList(Node).init(allocator), .edges = std.ArrayList(DirectedEdge).init(allocator) };
        return ret;
    }

    pub fn deinit(self: Self) void {
        self.nodes.deinit();
        self.edges.deinit();
    }

    pub fn constant(self: *Self, value: f32) !NodeHandle {
        var handle: NodeHandle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "constant", .op = .Constant, .value = value, .grad = 0.0 });
        return handle;
    }

    pub fn add(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        var handle: NodeHandle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "add", .op = .Addition, .value = null, .grad = 0.0 });
        try self.edges.append(DirectedEdge{ .from = a, .to = handle });
        try self.edges.append(DirectedEdge{ .from = b, .to = handle });
        return handle;
    }

    pub fn sub(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        // TODO: This can reuse Mul and Add. add(self, mul(-1, other))
        var handle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "sub", .op = .Subtraction, .value = null, .grad = 0.0 });
        try self.edges.append(DirectedEdge{ .from = a, .to = handle });
        try self.edges.append(DirectedEdge{ .from = b, .to = handle });
        return handle;
    }

    pub fn mul(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        var handle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "mul", .op = .Multiplication, .value = null, .grad = 0.0 });
        try self.edges.append(DirectedEdge{ .from = a, .to = handle });
        try self.edges.append(DirectedEdge{ .from = b, .to = handle });
        return handle;
    }

    pub fn tanh(self: *Self, a: NodeHandle) !NodeHandle {
        var handle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "tanh", .op = .TanH, .value = null, .grad = 0.0 });
        try self.edges.append(DirectedEdge{ .from = a, .to = handle });
        return handle;
    }

    const ResolveMode = enum { Forward, Backward };

    pub fn resolveNode(self: *Self, node_handle: NodeHandle, comptime mode: ResolveMode) !void {
        // TODO: Can perhaps wreplace this with a call to ensureTopologicalOrder,
        // that will resort the Nodes and how they are stored in mem?
        var sorted_nodes = try self.toposort_dfs(node_handle);
        defer sorted_nodes.deinit();

        switch (mode) {
            inline .Forward => {
                std.log.info("Performing forward pass in DAG", .{});

                // Traverse nodes in reverse-topological order for the forward pass.
                std.mem.reverse(usize, sorted_nodes.items);
            },
            inline else => {
                std.log.info("Performing backward pass in DAG", .{});
                self.nodes.items[node_handle].grad = 1.0;
            },
        }

        for (sorted_nodes.items) |current_node_handle| {
            var current_node = &self.nodes.items[current_node_handle];
            std.log.debug("EVALUATING {s}, {?}", .{ current_node.name, current_node });

            if (current_node.op == .Constant) continue;

            // Limit to 2 incoming edges to a given node.
            // TODO: Quicker lookup for incoming nodes?
            // Pre-processing step to prepare a lookup table?
            var in_a: ?*Node = null;
            var in_b: ?*Node = null;
            for (self.edges.items) |edge| {
                if (edge.to != current_node_handle) continue;
                const incomming_node = &self.nodes.items[edge.from];
                std.log.debug("Incoming: {s} -> {s}", .{ incomming_node.name, current_node.name });
                if (in_a == null) {
                    in_a = incomming_node;
                } else if (in_b == null) {
                    in_b = incomming_node;
                } else {
                    std.log.err("One too many incoming nodes to {s} -> {s}: {?}: {?}", .{ incomming_node.name, current_node.name, incomming_node, current_node });
                    assert(false); // We only allow 2 incoming edges to a node.
                }
            }

            // TODO: This check should not be performed for unary ops.
            if (current_node.op != .Constant and current_node.op != .TanH and (in_a == null or in_b == null)) {
                std.debug.print("Not enough inputs to Op node {s}: {?}", .{ current_node.name, current_node });
                assert(false); // Not enough inputs to op node.
            }

            // TODO: Clean this up.
            switch (mode) {
                .Forward => {
                    switch (current_node.op) {
                        .Constant => {
                            // do nothing, value should already be set.
                        },
                        .Addition => {
                            OpAddition.forward(current_node, @ptrCast(*const Node, in_a.?), @ptrCast(*const Node, in_b.?));
                        },
                        .Subtraction => {
                            OpSubtraction.forward(current_node, @ptrCast(*const Node, in_a.?), @ptrCast(*const Node, in_b.?));
                        },
                        .Multiplication => {
                            OpMultiplication.forward(current_node, @ptrCast(*const Node, in_a.?), @ptrCast(*const Node, in_b.?));
                        },
                        .TanH => {
                            OpTanH.forward(current_node, @ptrCast(*const Node, in_a.?));
                        },
                    }
                },
                .Backward => {
                    switch (current_node.op) {
                        .Constant => {
                            // do nothing, value should already be set.
                        },
                        .Addition => {
                            OpAddition.backward(current_node, in_a.?, in_b.?);
                        },
                        .Subtraction => {
                            OpSubtraction.backward(current_node, in_a.?, in_b.?);
                        },
                        .Multiplication => {
                            OpMultiplication.backward(current_node, in_a.?, in_b.?);
                        },
                        .TanH => {
                            OpTanH.backward(current_node, in_a.?);
                        },
                    }
                },
            }

            std.log.debug("DONE EVALUATING {s}, {?}", .{ current_node.name, current_node });
        }
    }

    pub fn toposort_dfs(self: *Self, source: NodeHandle) !ArrayList(NodeHandle) {
        // TODO: Assert acyclic.

        var stack = ArrayList(NodeHandle).init(self.allocator);
        defer stack.deinit();

        var visited = try ArrayList(bool).initCapacity(self.allocator, self.nodes.items.len);
        for (self.nodes.items) |_| {
            try visited.append(false);
        }
        defer visited.deinit();

        var sorted_nodes = ArrayList(NodeHandle).init(self.allocator);

        try stack.append(source);
        while (stack.items.len > 0) {
            var current = stack.pop();
            if (visited.items[current]) continue;
            visited.items[current] = true;

            try sorted_nodes.append(current);

            for (self.edges.items) |edge| {
                if (edge.to != current) {
                    continue;
                }
                try stack.append(edge.from);
            }
        }

        return sorted_nodes;
    }

    const Op = enum { Constant, Addition, Subtraction, Multiplication, TanH };
    const NodeHandle = u64;
    const DirectedEdge = struct { from: NodeHandle, to: NodeHandle };
    const Node = struct {
        name: []const u8,
        op: Op,
        value: ?f32, // to be tensor?
        grad: f32, // to be tensor?
    };

    const OpInterface = struct {
        // Limited to only 1 other value in
        forward: *const fn (*f32, *const f32) void,
        backward: *const fn (*f32, *const f32) void,
    };
};

const OpAddition = struct {
    fn forward(out: *DAG.Node, in_a: *const DAG.Node, in_b: *const DAG.Node) void {
        out.value = in_a.value.? + in_b.value.?;
    }

    fn backward(out: *DAG.Node, in_a: *DAG.Node, in_b: *DAG.Node) void {
        in_a.grad += 1.0 * out.grad;
        in_b.grad += 1.0 * out.grad;
    }
};
test "OpAddition forward/backward" {
    var out = DAG.Node{ .name = "", .op = .Addition, .value = null, .grad = 0.5 };
    var in_a = DAG.Node{ .name = "", .op = .Constant, .value = 2.0, .grad = 0.0 };
    var in_b = DAG.Node{ .name = "", .op = .Constant, .value = 3.0, .grad = 0.0 };
    OpAddition.forward(&out, &in_a, &in_b);
    try std.testing.expectEqual(out.value, 5.0);

    OpAddition.backward(&out, &in_a, &in_b);
    try std.testing.expectEqual(out.grad, 0.5);
    try std.testing.expectEqual(in_a.grad, 0.5);
    try std.testing.expectEqual(in_b.grad, 0.5);
}

const OpSubtraction = struct {
    fn forward(out: *DAG.Node, in_a: *const DAG.Node, in_b: *const DAG.Node) void {
        out.value = in_a.value.? - in_b.value.?;
    }

    fn backward(out: *DAG.Node, in_a: *DAG.Node, in_b: *DAG.Node) void {
        in_a.grad += 1.0 * out.grad;
        in_b.grad += 1.0 * out.grad;
    }
};
test "OpSubtraction forward/backward" {
    var out = DAG.Node{ .name = "", .op = .Addition, .value = null, .grad = 0.5 };
    var in_a = DAG.Node{ .name = "", .op = .Constant, .value = 2.0, .grad = 0.0 };
    var in_b = DAG.Node{ .name = "", .op = .Constant, .value = 3.0, .grad = 0.0 };
    OpSubtraction.forward(&out, &in_a, &in_b);
    try std.testing.expectEqual(out.value, -1.0);

    OpSubtraction.backward(&out, &in_a, &in_b);
    try std.testing.expectEqual(out.grad, 0.5);
    try std.testing.expectEqual(in_a.grad, 0.5);
    try std.testing.expectEqual(in_b.grad, 0.5);
}

const OpMultiplication = struct {
    fn forward(out: *DAG.Node, in_a: *const DAG.Node, in_b: *const DAG.Node) void {
        out.value = in_a.value.? * in_b.value.?;
    }

    fn backward(out: *DAG.Node, in_a: *DAG.Node, in_b: *DAG.Node) void {
        in_a.grad += in_b.value.? * out.grad;
        in_b.grad += in_a.value.? * out.grad;
    }
};
test "OpMultiplication forward/backward" {
    var out = DAG.Node{ .name = "", .op = .Multiplication, .value = null, .grad = 1.0 };
    var in_a = DAG.Node{ .name = "", .op = .Constant, .value = 2.0, .grad = 0.0 };
    var in_b = DAG.Node{ .name = "", .op = .Constant, .value = 3.0, .grad = 0.0 };
    OpMultiplication.forward(&out, &in_a, &in_b);
    try std.testing.expectEqual(out.value, 6.0);

    OpMultiplication.backward(&out, &in_a, &in_b);
    try std.testing.expectEqual(out.grad, 1.0);
    try std.testing.expectEqual(in_a.grad, 3.0);
    try std.testing.expectEqual(in_b.grad, 2.0);
}

const OpTanH = struct {
    fn forward(out: *DAG.Node, in_a: *const DAG.Node) void {
        const v = in_a.value.?;
        out.value = (std.math.exp(2.0 * v) - 1) / (std.math.exp(2.0 * v) + 1);
    }

    fn backward(out: *DAG.Node, in_a: *DAG.Node) void {
        const t = out.value.?;
        in_a.grad += (1.0 - t * t) * out.grad;
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var graph = DAG.init(allocator);
    var a = try graph.constant(1);
    var b = try graph.constant(2);
    var sum = try graph.add(a, b);
    var c = try graph.constant(6);
    var sub = try graph.sub(sum, c);
    try graph.resolveNode(sub, .Forward);
    std.debug.print("Value of sub node: {?}\n", .{graph.nodes.items[sub].value});

    // TODOs:
    // - Move backing data store to a Tensor
    // - Split out the ops from the resolve function...
    // - Get inspiration from ECS videos on how to get the handles..

    // var device_cpu = TensorDeviceCPU.init(std.heap.page_allocator);
    // defer device_cpu.deinit();

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

test "dag test forward/backward f = (1+2) - 6" {
    var graph = DAG.init(std.testing.allocator);
    defer graph.deinit();

    // f = (1+2) - 6
    var a = try graph.constant(1.0);
    var b = try graph.constant(2.0);
    var sum = try graph.add(a, b);
    var c = try graph.constant(6.0);
    var sub = try graph.sub(sum, c);
    var out_node_handle = sub;

    // Forward
    try graph.resolveNode(out_node_handle, .Forward);
    try std.testing.expectEqual(graph.nodes.items[out_node_handle].value, -3.0);

    // Backward
    // try graph.resolveNodeBackward(out_node_handle);

    // MPS does it like this:
    // torch does it like this:
    //  logit1 = torch.Tensor([1.0]).double() ; logit4.requires_grad = True
    //  ...
    //  logits = [logit1, logit2, logit3, logit4]
    //  probs = softmax(logits)
    //  loss = -probs[3].log()
    //  loss.backward()
}

test "dag test forward/backward f = tanh(2*(-3) + (0*1) + 6.7)" {
    var graph = DAG.init(std.testing.allocator);
    defer graph.deinit();

    // f = tanh(2*(-3) + (0*1) + 6.7)
    var a = try graph.constant(2.0);
    var b = try graph.constant(-3.0);
    var mul1 = try graph.mul(a, b);
    var c = try graph.constant(0.0);
    var d = try graph.constant(1.0);
    var mul2 = try graph.mul(c, d);
    var sum1 = try graph.add(mul1, mul2);
    var e = try graph.constant(6.8813735870195432);
    var sum2 = try graph.add(sum1, e);
    var tanh = try graph.tanh(sum2);
    var out_node_handle = tanh;

    // Forward
    try graph.resolveNode(out_node_handle, .Forward);
    try std.testing.expectApproxEqAbs(graph.nodes.items[out_node_handle].value.?, 0.7071, 0.001);

    // Backward
    try graph.resolveNode(out_node_handle, .Backward);
    try std.testing.expectApproxEqAbs(graph.nodes.items[c].grad, 0.5, 0.001);
    try std.testing.expectApproxEqAbs(graph.nodes.items[d].grad, 0.0, 0.001);
    try std.testing.expectApproxEqAbs(graph.nodes.items[a].grad, -1.5, 0.001);
    try std.testing.expectApproxEqAbs(graph.nodes.items[b].grad, 1.0, 0.001);
}

test "tensor test" {
    var device_cpu = TensorDeviceCPU.init(std.testing.allocator);
    defer device_cpu.deinit();

    var tensor_a = try device_cpu.createTensor(TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 1, 1 },
    });
    device_cpu.setTensorData(tensor_a, &[_]f32{1.0});

    var tensor_b = try device_cpu.createTensor(TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 1, 1 },
    });
    device_cpu.setTensorData(tensor_b, &[_]f32{2.0});

    device_cpu.tensorOpAddition(tensor_a, tensor_b);
    var values = device_cpu.tensorValues(tensor_a);
    try std.testing.expectEqual(values[0], 3.0);
}
