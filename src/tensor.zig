const builtin = @import("builtin");
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

// TODO: SIMD:
//    - numpy broadcasting
//    - Different data types

pub const TensorDescriptor = struct {
    dimensions_sizes: []const usize,
    // data_type:
};
pub const Tensor = struct {
    descriptor: TensorDescriptor,
    handle: usize,
};

// TODO: Consider also doing a SIMD, BLAS or CUDA version to compare against
/// Naive CPU implementation of a tensor backend.
pub const TensorDeviceCPU = struct {
    const Self = @This();

    const TensorCPU = struct {
        elements: []f32,
    };

    allocator: Allocator,
    backing_tensors: ArrayList(TensorCPU),

    pub fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator, .backing_tensors = ArrayList(TensorCPU).init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        for (self.backing_tensors.items) |tensor| {
            self.allocator.free(tensor.elements);
        }
        self.backing_tensors.deinit();
    }

    pub fn createTensor(self: *Self, desc: TensorDescriptor) !Tensor {
        var num_elements: usize = 1;
        for (desc.dimensions_sizes) |dim| {
            assert(dim > 0); // Tensor dimension sizes must be > 0
            num_elements *= dim;
        }

        var elements = try self.allocator.alloc(f32, num_elements);
        @memset(elements, 0.0);
        var new_tensor_handle = self.backing_tensors.items.len;
        try self.backing_tensors.append(TensorCPU{ .elements = elements });
        return Tensor{ .handle = new_tensor_handle, .descriptor = desc };
    }

    pub fn createTensorScalar(self: *Self, value: f32) !Tensor {
        var tensor = try self.createTensor(TensorDescriptor{
            .dimensions_sizes = &[_]usize{1},
        });
        self.setTensorData(tensor, &[_]f32{value});
        return tensor;
    }

    pub fn setTensorData(self: *Self, tensor: Tensor, data: []const f32) void {
        var backing_tensor = self.backing_tensors.items[tensor.handle];
        assert(productOfElements(tensor.descriptor.dimensions_sizes) == data.len);
        std.mem.copy(f32, backing_tensor.elements, data);
    }

    pub fn tensorOpSet(self: *Self, tensor_dst: Tensor, tensor_src: Tensor) void {
        var backing_tensor_dst = self.backing_tensors.items[tensor_dst.handle];
        var backing_tensor_src = self.backing_tensors.items[tensor_src.handle];
        assert(std.mem.eql(usize, tensor_dst.descriptor.dimensions_sizes, tensor_src.descriptor.dimensions_sizes));
        std.mem.copy(f32, backing_tensor_dst.elements, backing_tensor_src.elements);
    }

    pub fn tensorOpSetScalar(self: *Self, tensor_dst: Tensor, value: f32) void {
        var backing_tensor_dst = self.backing_tensors.items[tensor_dst.handle];
        @memset(backing_tensor_dst.elements, value);
    }

    /// a += b (element-wise)
    pub fn tensorOpAddition(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements, 0..) |_, i| {
            backing_tensor_a.elements[i] += backing_tensor_b.elements[i];
        }
    }

    /// a -= b (element-wise)
    pub fn tensorOpSubtraction(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements, 0..) |_, i| {
            backing_tensor_a.elements[i] -= backing_tensor_b.elements[i];
        }
    }

    /// a *= b (element-wise)
    pub fn tensorOpMultiplication(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements, 0..) |_, i| {
            backing_tensor_a.elements[i] *= backing_tensor_b.elements[i];
        }
    }

    /// a += b * c (element-wise)
    pub fn tensorOpMultiplicationAccumulation(self: *Self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        var backing_tensor_c = self.backing_tensors.items[tensor_c.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_c.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements, 0..) |_, i| {
            backing_tensor_a.elements[i] += backing_tensor_b.elements[i] * backing_tensor_c.elements[i];
        }
    }

    /// a /= b (element-wise)
    pub fn tensorOpDivision(self: *Self, tensor_a: Tensor, tensor_b: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements, 0..) |_, i| {
            backing_tensor_a.elements[i] /= backing_tensor_b.elements[i];
        }
    }

    /// a = exp(a) (element-wise)
    pub fn tensorOpExp(self: *Self, tensor_a: Tensor) void {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        for (backing_tensor_a.elements, 0..) |_, i| {
            backing_tensor_a.elements[i] = std.math.exp(backing_tensor_a.elements[i]);
        }
    }

    /// a = tanh(a) (element-wise)
    pub fn tensorOpTanh(self: *Self, tensor_a: Tensor) void {
        // TODO: Can be constructed from constituent ops?
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        for (backing_tensor_a.elements, 0..) |_, i| {
            const v = backing_tensor_a.elements[i];
            backing_tensor_a.elements[i] = (std.math.exp(2.0 * v) - 1) / (std.math.exp(2.0 * v) + 1);
        }
    }

    /// a += (1- b*b) * (element-wise)
    pub fn tensorOpTanhBackward(self: *Self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor) void {
        // TODO: Can be constructed from constituent ops?
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var backing_tensor_b = self.backing_tensors.items[tensor_b.handle];
        var backing_tensor_c = self.backing_tensors.items[tensor_c.handle];
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_b.descriptor.dimensions_sizes));
        assert(std.mem.eql(usize, tensor_a.descriptor.dimensions_sizes, tensor_c.descriptor.dimensions_sizes));
        for (backing_tensor_a.elements, 0..) |_, i| {
            const t = backing_tensor_b.elements[i];
            backing_tensor_a.elements[i] += (1.0 - t * t) * backing_tensor_c.elements[i];
        }
    }

    pub fn tensorValues(self: *Self, tensor_a: Tensor) []f32 {
        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        return backing_tensor_a.elements;
    }

    pub fn tensorValue(self: *Self, tensor_a: Tensor, index: []const usize) f32 {
        const dimensions = tensor_a.descriptor.dimensions_sizes;
        if (index.len != dimensions.len) {
            std.log.err("Dimension mismatch with index {any} and tensor {any}!", .{ index, dimensions });
            assert(false); // Mismatched index to tensor dimension
        }

        var backing_tensor_a = self.backing_tensors.items[tensor_a.handle];
        var actual_index: usize = 0;
        var stride: usize = 1;
        for (index, 0..) |_, i| {
            var i_reversed = index.len - i - 1;
            var index_el = index[i_reversed];
            actual_index += index_el * stride;
            stride *= dimensions[i_reversed];

            if (builtin.mode != .ReleaseFast and !(index_el < dimensions[i_reversed])) {
                std.log.err("Out of bounds access {any} in tensor with dimension {any}!", .{ index, dimensions });
                assert(false); // Out of bounds access in Tensor
            }
        }
        return backing_tensor_a.elements[actual_index];
    }
};

fn productOfElements(elements: []const usize) usize {
    var product: usize = 1;
    for (elements) |element| {
        product *= element;
    }
    return product;
}

test {
    _ = @import("tensor_test.zig");
}
