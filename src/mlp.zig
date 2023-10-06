const std = @import("std");
const Allocator = std.mem.Allocator;

const Dag = @import("dag.zig").Dag;

// MLP is constrained to 1d vectors as input and output (sequential)
pub const Mlp = struct {
    const Self = @This();
    allocator: Allocator,
    dag_graph: *Dag,
    // layer_stack: *Dag, // TODO

    const LayerDescriptor = struct {
        layer_type: LayerType,
        num_neurons: usize,
    };
    const Layer = struct {};

    pub fn init(allocator: Allocator, graph: *Dag) @This() {
        return Self{ .allocator = allocator, .dag_graph = graph };
    }

    pub fn add(self: *Self, desc: LayerDescriptor) !void {
        _ = self;
        _ = desc;

        // for a fully connected layer we'll have to:
        // Initialize weight matrix and a bias vector
        // https://medium.com/analytics-vidhya/simple-cnn-using-numpy-part-iv-back-propagation-through-fully-connected-layers-c5035d678307
    }
};

const LayerType = enum {
    input,
    fully_connected,
};

// Input layer: Shape of the input tensor is required
// Dense layer: (needs to know number of neurons, number of inputs should be known from previous layer)
// output = activation(dot(input, kernel) + bias)
// element wise activation,
// kernel is weights matrix (created by layer),
// bias is a bias vector (created by layer)

test "Mlp" {
    const TensorDevice = @import("tensor.zig").TensorDeviceCPU;
    var tensor_dev = TensorDevice.init(std.testing.allocator);
    defer tensor_dev.deinit();
    var graph = Dag.init(std.testing.allocator, &tensor_dev);
    defer graph.deinit();

    var mlp = Mlp.init(std.testing.allocator, &graph);
    _ = mlp;
}
