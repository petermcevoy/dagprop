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

    pub fn constant(self: *Self, value: i32) !NodeHandle {
        var handle: NodeHandle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "test", .op = .Constant, .value = value });
        return handle;
    }

    pub fn add(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        var handle: NodeHandle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "add", .op = .Addition, .value = null });
        try self.edges.append(DirectedEdge{ .from = a, .to = handle });
        try self.edges.append(DirectedEdge{ .from = b, .to = handle });
        return handle;
    }

    pub fn sub(self: *Self, a: NodeHandle, b: NodeHandle) !NodeHandle {
        var handle = self.nodes.items.len;
        try self.nodes.append(Node{ .name = "sub", .op = .Subtraction, .value = null });
        try self.edges.append(DirectedEdge{ .from = a, .to = handle });
        try self.edges.append(DirectedEdge{ .from = b, .to = handle });
        return handle;
    }

    pub fn resolveNode(self: *Self, node_handle: NodeHandle) !void {
        var sorted_nodes = try self.toposort_dfs(node_handle);
        defer sorted_nodes.deinit();

        var n = sorted_nodes.items.len;
        while (n > 0) : (n -= 1) {
            var i = n - 1;
            var current_node_handle = sorted_nodes.items[i];
            var current_node = &self.nodes.items[current_node_handle];
            std.debug.print("EVALUATING {s}, {?}\n", .{ current_node.name, current_node });

            switch (current_node.op) {
                .Constant => {
                    // do nothing, value should already be set.
                },
                .Addition => {
                    // Get the input edges to this node, they should already be evaluated.
                    var sum: i32 = 0;
                    for (self.edges.items) |edge| {
                        if (edge.to != current_node_handle) continue;
                        std.debug.print("\t {?}\n", .{edge});
                        sum += self.nodes.items[edge.from].value.?;
                    }
                    current_node.value = sum;
                },
                .Subtraction => {
                    // Get the input edges to this node, they should already be evaluated.
                    var result: ?i32 = null;
                    for (self.edges.items) |edge| {
                        if (edge.to != current_node_handle) continue;
                        var value = self.nodes.items[edge.from].value.?;
                        if (result == null) {
                            result = value;
                        } else {
                            result = result.? - value;
                        }
                        std.debug.print("s {?}\n", .{result});
                    }
                    current_node.value = result;
                },
            }
            std.debug.print("DONE EVALUATING {s}, {?}\n", .{ current_node.name, current_node });
        }

        std.debug.print("Sorted nodes:\n", .{});
        for (sorted_nodes.items) |current_node_handle| {
            var node = self.nodes.items[current_node_handle];
            std.debug.print("{?}\n", .{node});
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
            std.debug.print("Visiting {s} ({})\n", .{ self.nodes.items[current].name, current });

            try sorted_nodes.append(current);

            for (self.edges.items) |edge| {
                std.debug.print("Checking edge {?}\n", .{edge});
                if (edge.to != current) {
                    continue;
                }
                try stack.append(edge.from);
            }
        }

        return sorted_nodes;
    }

    const Op = enum { Constant, Addition, Subtraction };
    const NodeHandle = u64;
    const DirectedEdge = struct { from: NodeHandle, to: NodeHandle };
    const Node = struct {
        name: []const u8,
        op: Op,
        value: ?i32, // to be tensor?
    };
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
    try graph.resolveNode(sub);
    std.debug.print("Value of sub node: {?}\n", .{graph.nodes.items[sub].value});

    // TODOs:
    // - Move backing data store to a Tensor
    // - Split out the ops from the resolve function...
    // - Get inspiration from ECS videos on how to get the handles..
    // - actual Backprop

    var device_cpu = TensorDeviceCPU.init(std.heap.page_allocator);
    defer device_cpu.deinit();

    var tensor = try device_cpu.createTensor(TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 256, 512 },
    });
    _ = tensor;

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
    std.debug.print("Values: {any}", .{values});

    // Goal 1 recreate micrograd
    //

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

test "dag test" {
    var graph = DAG.init(std.testing.allocator);
    defer graph.deinit();
    var a = try graph.constant(1);
    var b = try graph.constant(2);
    var sum = try graph.add(a, b);
    var c = try graph.constant(6);
    var sub = try graph.sub(sum, c);
    try graph.resolveNode(sub);
    try std.testing.expectEqual(graph.nodes.items[sub].value, -3);
}

test "tensor test" {
    var device_cpu = TensorDeviceCPU.init(std.testing.allocator);
    defer device_cpu.deinit();

    var tensor = try device_cpu.createTensor(TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 256, 512 },
    });
    _ = tensor;
}
