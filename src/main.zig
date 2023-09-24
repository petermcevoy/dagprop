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

        var elements = try self.allocator.alloc(f32, num_elements);
        std.mem.set(f32, elements, 0.0);
        var new_tensor_handle = self.backing_tensors.items.len;
        try self.backing_tensors.append(TensorCPU{ .elements = elements });
        return Tensor{ .handle = new_tensor_handle, .descriptor = desc };
    }

    fn createTensorScalar(self: *Self, value: f32) !Tensor {
        var tensor = try self.createTensor(TensorDescriptor{
            .dimensions_sizes = &[_]usize{1},
        });
        self.setTensorData(tensor, &[_]f32{value});
        return tensor;
    }

    fn setTensorData(self: *Self, tensor: Tensor, data: []const f32) void {
        var backing_tensor = self.backing_tensors.items[tensor.handle];
        assert(productOfElements(tensor.descriptor.dimensions_sizes) == data.len);
        std.mem.copy(f32, backing_tensor.elements, data);
    }

    fn tensorOpSet(self: *Self, tensor_dst: Tensor, tensor_src: Tensor) void {
        var backing_tensor_dst = self.backing_tensors.items[tensor_dst.handle];
        var backing_tensor_src = self.backing_tensors.items[tensor_src.handle];
        assert(std.mem.eql(usize, tensor_dst.descriptor.dimensions_sizes, tensor_src.descriptor.dimensions_sizes));
        std.mem.copy(f32, backing_tensor_dst.elements, backing_tensor_src.elements);
    }

    fn tensorOpSetScalar(self: *Self, tensor_dst: Tensor, value: f32) void {
        var backing_tensor_dst = self.backing_tensors.items[tensor_dst.handle];
        std.mem.set(f32, backing_tensor_dst.elements, value);
    }

    fn tensorOpAddition(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] += backing_tensor_b.elements[i];
        }
    }

    fn tensorOpSubtraction(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] -= backing_tensor_b.elements[i];
        }
    }

    fn tensorOpMultiplication(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] *= backing_tensor_b.elements[i];
        }
    }

    fn tensorOpMultiplicationAccumulation(self: *Self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        var backing_tensor_c = self.backing_tensors.items[tensor_c.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_c.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] += backing_tensor_b.elements[i] * backing_tensor_c.elements[i];
        }
    }

    fn tensorOpDivision(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] /= backing_tensor_b.elements[i];
        }
    }

    fn tensorOpExp(self: *Self, tensor_a: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        for (backing_tensor_a.elements) |_, i| {
            backing_tensor_a.elements[i] = std.math.exp(backing_tensor_a.elements[i]);
        }
    }

    fn tensorOpTanh(self: *Self, tensor_a: Tensor) void { // TODO: more generic way to do this?
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        for (backing_tensor_a.elements) |_, i| {
            const v = backing_tensor_a.elements[i];
            backing_tensor_a.elements[i] = (std.math.exp(2.0 * v) - 1) / (std.math.exp(2.0 * v) + 1);
        }
    }

    fn tensorOpTanhBackward(self: *Self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor) void { // TODO: more generic way to do this?
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        var backing_tensor_c = self.backing_tensors.items[tensor_c.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_c.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements) |_, i| {
            const t = backing_tensor_b.elements[i];
            backing_tensor_a.elements[i] += (1.0 - t * t) * backing_tensor_c.elements[i];
        }
    }

    fn tensorValues(self: *Self, tensor_a: Tensor) []f32 {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        return backing_tensor_a.elements;
    }

    fn tensorValue(self: *Self, tensor_a: Tensor, index: usize) f32 {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        return backing_tensor_a.elements[index];
    }
};

// TODO: Support more device backbends, simd, Metal performance shaders, cuda?
const TensorDevice = TensorDeviceCPU;

const TensorDescriptor = struct {
    dimensions_sizes: []const usize,
    // data_type: ,
};
const Tensor = struct {
    descriptor: TensorDescriptor,
    handle: usize,
};

const DAG = struct {
    const Self = @This();
    nodes: ArrayList(Node),
    edges: ArrayList(DirectedEdge),
    allocator: Allocator,
    tensor_device: *TensorDevice,

    pub fn init(allocator: Allocator, tensor_device: *TensorDevice) @This() {
        return Self{
            .allocator = allocator,
            .nodes = std.ArrayList(Node).init(allocator),
            .edges = std.ArrayList(DirectedEdge).init(allocator),
            .tensor_device = tensor_device,
        };
    }

    pub fn deinit(self: Self) void {
        self.nodes.deinit();
        self.edges.deinit();
    }

    fn addNode(self: *Self, name: []const u8, op: Op, value: ?Tensor, inputs: []NodeHandle) !NodeHandle {
        var handle: NodeHandle = self.nodes.items.len;

        var tensor_value: Tensor = undefined;
        var tensor_grad: Tensor = undefined;
        {
            var dimensions: []const usize = undefined;
            if (value != null) {
                dimensions = value.?.descriptor.dimensions_sizes;
            } else {
                for (inputs) |input| {
                    const backing_input_tensor = self.nodes.items[input];
                    dimensions = backing_input_tensor.grad.descriptor.dimensions_sizes;
                }
            }
            tensor_grad = try self.tensor_device.createTensor(.{ .dimensions_sizes = dimensions });

            if (value != null) {
                tensor_value = value.?;
            } else {
                tensor_value = try self.tensor_device.createTensor(.{ .dimensions_sizes = dimensions });
            }
        }

        try self.nodes.append(Node{
            .name = name,
            .op = op,
            .value = tensor_value,
            .grad = tensor_grad,
        });

        // Add input edges to node
        for (inputs) |input| {
            try self.edges.append(DirectedEdge{ .from = input, .to = handle });
        }

        return handle;
    }

    pub fn constantScalar(self: *Self, value: f32) !NodeHandle {
        var tensor_value = try self.tensor_device.createTensorScalar(value);
        return try self.addNode("constant", .Constant, tensor_value, &[0]NodeHandle{});
    }

    pub fn constant(self: *Self, value: Tensor) !NodeHandle {
        return try self.addNode("constant", .Constant, value, &[0]NodeHandle{});
    }

    pub fn add(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        return try self.addNode("add", Op.from(OpAddition), null, &[_]NodeHandle{ a, b });
    }

    pub fn sub(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        // TODO: This can be replaced by Mul and Add. add(self, mul(-1, other))
        return try self.addNode("sub", Op.from(OpSubtraction), null, &[_]NodeHandle{ a, b });
    }

    pub fn mul(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        return try self.addNode("mul", Op.from(OpMultiplication), null, &[_]NodeHandle{ a, b });
    }

    pub fn tanh(self: *Self, a: NodeHandle) !NodeHandle {
        return try self.addNode("tanh", Op.from(OpTanH), null, &[_]NodeHandle{a});
    }

    const ResolveMode = enum { Forward, Backward };

    pub fn resolveNode(self: *Self, node_handle: NodeHandle, comptime mode: ResolveMode) !void {
        // TODO: Can perhaps replace this with a call to ensureTopologicalOrder,
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
                self.tensor_device.tensorOpSetScalar(self.nodes.items[node_handle].grad, 1.0);
            },
        }

        for (sorted_nodes.items) |current_node_handle| {
            var current_node = &self.nodes.items[current_node_handle];
            std.log.debug("EVALUATING {s}, {?}", .{ current_node.name, current_node });

            if (current_node.op == .Constant) continue;

            // Limit to 2 incoming edges to a given node.
            // TODO: Quicker lookup for incoming nodes?
            // TODO: Pre-processing step to prepare a lookup table?
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

            if ((current_node.op == .Unary and (in_a == null)) or
                (current_node.op == .Binary and (in_a == null or in_b == null)))
            {
                std.log.err("Not enough inputs to Op node {s}: {?}", .{ current_node.name, current_node });
                assert(false); // Not enough inputs to op node.
            }

            switch (current_node.op) {
                .Constant => {
                    // do nothing
                },
                .Unary => |unary_op| {
                    switch (mode) {
                        .Forward => {
                            unary_op.forward(self.tensor_device, current_node, @ptrCast(*const Node, in_a.?));
                        },
                        .Backward => {
                            unary_op.backward(self.tensor_device, current_node, @ptrCast(*Node, in_a.?));
                        },
                    }
                },
                .Binary => |binary_op| {
                    switch (mode) {
                        .Forward => {
                            binary_op.forward(self.tensor_device, current_node, @ptrCast(*const Node, in_a.?), @ptrCast(*const Node, in_b.?));
                        },
                        .Backward => {
                            binary_op.backward(self.tensor_device, current_node, @ptrCast(*Node, in_a.?), @ptrCast(*Node, in_b.?));
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

    const NodeHandle = u64;
    const DirectedEdge = struct { from: NodeHandle, to: NodeHandle };
};

const Node = struct {
    name: []const u8,
    op: Op,
    value: ?Tensor,
    grad: Tensor,
};

const OpAddition = struct {
    fn forward(dev: *TensorDevice, out: *Node, in_a: *const Node, in_b: *const Node) void {
        dev.tensorOpAddition(out.value.?, in_a.value.?);
        dev.tensorOpAddition(out.value.?, in_b.value.?);
    }

    fn backward(dev: *TensorDevice, out: *Node, in_a: *Node, in_b: *Node) void {
        dev.tensorOpAddition(in_a.grad, out.grad);
        dev.tensorOpAddition(in_b.grad, out.grad);
    }
};
test "OpAddition forward/backward" {
    var dev = TensorDevice.init(std.testing.allocator);
    defer dev.deinit();

    var out = Node{ .name = "", .op = Op.from(OpAddition), .value = try dev.createTensorScalar(0.0), .grad = try dev.createTensorScalar(0.5) };
    var in_a = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(2.0), .grad = try dev.createTensorScalar(0.0) };
    var in_b = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(3.0), .grad = try dev.createTensorScalar(0.0) };
    OpAddition.forward(&dev, &out, &in_a, &in_b);
    try std.testing.expectEqual(dev.tensorValue(out.value.?, 0), 5.0);

    OpAddition.backward(&dev, &out, &in_a, &in_b);
    try std.testing.expectEqual(dev.tensorValue(out.grad, 0), 0.5);
    try std.testing.expectEqual(dev.tensorValue(in_a.grad, 0), 0.5);
    try std.testing.expectEqual(dev.tensorValue(in_b.grad, 0), 0.5);
}

const OpSubtraction = struct {
    fn forward(dev: *TensorDevice, out: *Node, in_a: *const Node, in_b: *const Node) void {
        dev.tensorOpSet(out.value.?, in_a.value.?);
        dev.tensorOpSubtraction(out.value.?, in_b.value.?);
    }

    fn backward(dev: *TensorDevice, out: *Node, in_a: *Node, in_b: *Node) void {
        dev.tensorOpAddition(in_a.grad, out.grad);
        dev.tensorOpAddition(in_b.grad, out.grad);
    }
};
test "OpSubtraction forward/backward" {
    var dev = TensorDeviceCPU.init(std.testing.allocator);
    defer dev.deinit();

    var out = Node{ .name = "", .op = Op.from(OpAddition), .value = try dev.createTensorScalar(0.0), .grad = try dev.createTensorScalar(0.5) };
    var in_a = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(2.0), .grad = try dev.createTensorScalar(0.0) };
    var in_b = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(3.0), .grad = try dev.createTensorScalar(0.0) };
    OpSubtraction.forward(&dev, &out, &in_a, &in_b);
    try std.testing.expectEqual(dev.tensorValue(out.value.?, 0), -1.0);

    OpSubtraction.backward(&dev, &out, &in_a, &in_b);
    try std.testing.expectEqual(dev.tensorValue(out.grad, 0), 0.5);
    try std.testing.expectEqual(dev.tensorValue(in_a.grad, 0), 0.5);
    try std.testing.expectEqual(dev.tensorValue(in_b.grad, 0), 0.5);
}

const OpMultiplication = struct {
    fn forward(dev: *TensorDevice, out: *Node, in_a: *const Node, in_b: *const Node) void { // TODO: Make node write access respect const qualifiers? New ConstNode type?
        dev.tensorOpSet(out.value.?, in_a.value.?);
        dev.tensorOpMultiplication(out.value.?, in_b.value.?);
    }

    fn backward(dev: *TensorDevice, out: *Node, in_a: *Node, in_b: *Node) void {
        dev.tensorOpMultiplicationAccumulation(in_a.grad, in_b.value.?, out.grad);
        dev.tensorOpMultiplicationAccumulation(in_b.grad, in_a.value.?, out.grad);
    }
};
test "OpMultiplication forward/backward" {
    var dev = TensorDeviceCPU.init(std.testing.allocator);
    defer dev.deinit();

    var out = Node{ .name = "", .op = Op.from(OpMultiplication), .value = try dev.createTensorScalar(0.0), .grad = try dev.createTensorScalar(1.0) };
    var in_a = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(2.0), .grad = try dev.createTensorScalar(0.0) };
    var in_b = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(3.0), .grad = try dev.createTensorScalar(0.0) };
    OpMultiplication.forward(&dev, &out, &in_a, &in_b);
    try std.testing.expectEqual(dev.tensorValue(out.value.?, 0), 6.0);

    OpMultiplication.backward(&dev, &out, &in_a, &in_b);
    try std.testing.expectEqual(dev.tensorValue(out.grad, 0), 1.0);
    try std.testing.expectEqual(dev.tensorValue(in_a.grad, 0), 3.0);
    try std.testing.expectEqual(dev.tensorValue(in_b.grad, 0), 2.0);
}

const OpTanH = struct {
    fn forward(dev: *TensorDevice, out: *Node, in_a: *const Node) void {
        dev.tensorOpSet(out.value.?, in_a.value.?);
        dev.tensorOpTanh(out.value.?);
    }

    fn backward(dev: *TensorDevice, out: *Node, in_a: *Node) void {
        dev.tensorOpTanhBackward(in_a.grad, out.value.?, out.grad);
    }
};

test "OpTanH forward/backward" {
    var dev = TensorDeviceCPU.init(std.testing.allocator);
    defer dev.deinit();

    var out = Node{ .name = "", .op = Op.from(OpTanH), .value = try dev.createTensorScalar(0.0), .grad = try dev.createTensorScalar(1.0) };
    var in_a = Node{ .name = "", .op = .Constant, .value = try dev.createTensorScalar(2.0), .grad = try dev.createTensorScalar(0.0) };

    OpTanH.forward(&dev, &out, &in_a);
    try std.testing.expectEqual(dev.tensorValue(out.value.?, 0), 0.9640275801);

    OpTanH.backward(&dev, &out, &in_a);
    try std.testing.expectEqual(dev.tensorValue(out.grad, 0), 1.0);
    try std.testing.expectEqual(dev.tensorValue(in_a.grad, 0), 0.0706508159);
}

const OpUnary = struct {
    forward: *const fn (dev: *TensorDevice, out: *Node, in_a: *const Node) void,
    backward: *const fn (dev: *TensorDevice, out: *Node, in_a: *Node) void,
};

const OpBinary = struct {
    forward: *const fn (dev: *TensorDevice, out: *Node, in_a: *const Node, in_b: *const Node) void,
    backward: *const fn (dev: *TensorDevice, out: *Node, in_a: *Node, in_b: *Node) void,
};

const Op = union(enum) {
    Constant,
    Unary: OpUnary,
    Binary: OpBinary,

    pub fn from(T: anytype) Op {
        const has_unary_forward = @TypeOf(T.forward) == fn (dev: *TensorDevice, out: *Node, in_a: *const Node) void;
        const has_unary_backward = @TypeOf(T.backward) == fn (dev: *TensorDevice, out: *Node, in_a: *Node) void;
        const has_binary_forward = @TypeOf(T.forward) == fn (dev: *TensorDevice, out: *Node, in_a: *const Node, *const Node) void;
        const has_binary_backward = @TypeOf(T.backward) == fn (dev: *TensorDevice, out: *Node, in_a: *Node, in_b: *Node) void;
        const is_unary = has_unary_forward and has_unary_backward;
        const is_binary = has_binary_forward and has_binary_backward;

        if (is_unary == is_binary) {
            @compileError("Type has incorrect forward/backward functions for it to be an Op: " ++ @typeName(T));
        }

        if (is_unary) return Op{ .Unary = OpUnary{ .forward = T.forward, .backward = T.backward } };
        if (is_binary) return Op{ .Binary = OpBinary{ .forward = T.forward, .backward = T.backward } };
    }
};

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
    std.debug.print("Value of sub node: {?}\n", .{graph.nodes.items[sub].value});

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

test "dag test forward/backward f = (1+2) - 6" {
    var tensor_dev = TensorDevice.init(std.testing.allocator);
    defer tensor_dev.deinit();

    var graph = DAG.init(std.testing.allocator, &tensor_dev);
    defer graph.deinit();

    // f = (1+2) - 6
    var a = try graph.constant(try tensor_dev.createTensorScalar(1.0));
    var b = try graph.constant(try tensor_dev.createTensorScalar(2.0));
    var sum = try graph.add(a, b);
    var c = try graph.constant(try tensor_dev.createTensorScalar(6.0));
    var sub = try graph.sub(sum, c);
    var out_node_handle = sub;

    // Forward
    try graph.resolveNode(out_node_handle, .Forward);
    try std.testing.expectEqual(tensor_dev.tensorValue(graph.nodes.items[out_node_handle].value.?, 0), -3.0);

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
    var tensor_dev = TensorDevice.init(std.testing.allocator);
    defer tensor_dev.deinit();

    var graph = DAG.init(std.testing.allocator, &tensor_dev);
    defer graph.deinit();

    // f = tanh(2*(-3) + (0*1) + 6.7)
    var a = try graph.constantScalar(2.0);
    var b = try graph.constantScalar(-3.0);
    var mul1 = try graph.mul(a, b);
    var c = try graph.constantScalar(0.0);
    var d = try graph.constantScalar(1.0);
    var mul2 = try graph.mul(c, d);
    var sum1 = try graph.add(mul1, mul2);
    var e = try graph.constantScalar(6.8813735870195432);
    var sum2 = try graph.add(sum1, e);
    var tanh = try graph.tanh(sum2);
    var out_node_handle = tanh;

    // Forward
    try graph.resolveNode(out_node_handle, .Forward);
    try std.testing.expectApproxEqAbs(tensor_dev.tensorValue(graph.nodes.items[out_node_handle].value.?, 0), 0.7071, 0.001);

    // Backward
    try graph.resolveNode(out_node_handle, .Backward);
    try std.testing.expectApproxEqAbs(tensor_dev.tensorValue(graph.nodes.items[c].grad, 0), 0.5, 0.001);
    try std.testing.expectApproxEqAbs(tensor_dev.tensorValue(graph.nodes.items[d].grad, 0), 0.0, 0.001);
    try std.testing.expectApproxEqAbs(tensor_dev.tensorValue(graph.nodes.items[a].grad, 0), -1.5, 0.001);
    try std.testing.expectApproxEqAbs(tensor_dev.tensorValue(graph.nodes.items[b].grad, 0), 1.0, 0.001);
}

test "tensor test" {
    var device_cpu = TensorDevice.init(std.testing.allocator);
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

    device_cpu.tensorOpMultiplication(tensor_a, tensor_b);
    values = device_cpu.tensorValues(tensor_a);
    try std.testing.expectEqual(values[0], 6.0);

    device_cpu.tensorOpDivision(tensor_a, tensor_b);
    values = device_cpu.tensorValues(tensor_a);
    try std.testing.expectEqual(values[0], 3.0);

    device_cpu.tensorOpExp(tensor_a);
    values = device_cpu.tensorValues(tensor_a);
    try std.testing.expectEqual(values[0], 20.0855369);
}
