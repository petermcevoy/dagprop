const std = @import("std");
const tensor = @import("tensor.zig");
const TensorDevice = tensor.TensorDeviceCPU;

test "tensor test scalar" {
    var device = TensorDevice.init(std.testing.allocator);
    defer device.deinit();

    var tensor_a = try device.createTensor(tensor.TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 1, 1 },
    });
    device.setTensorData(tensor_a, &[_]f32{1.0});

    var tensor_b = try device.createTensor(tensor.TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 1, 1 },
    });
    device.setTensorData(tensor_b, &[_]f32{2.0});

    device.tensorOpAddition(tensor_a, tensor_b);
    try std.testing.expectEqual(device.tensorValues(tensor_a)[0], 3.0);

    device.tensorOpMultiplication(tensor_a, tensor_b);
    try std.testing.expectEqual(device.tensorValues(tensor_a)[0], 6.0);

    device.tensorOpDivision(tensor_a, tensor_b);
    try std.testing.expectEqual(device.tensorValues(tensor_a)[0], 3.0);

    device.tensorOpExp(tensor_a);
    try std.testing.expectEqual(device.tensorValues(tensor_a)[0], 20.0855369);
}

test "tensor test 2x2" {
    var device = TensorDevice.init(std.testing.allocator);
    defer device.deinit();

    var tensor_a = try device.createTensor(tensor.TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 2, 2 },
    });
    device.setTensorData(tensor_a, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });

    var tensor_b = try device.createTensor(tensor.TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 2, 2 },
    });
    device.setTensorData(tensor_b, &[_]f32{ 2.0, 4.0, 6.0, 8.0 });

    device.tensorOpAddition(tensor_a, tensor_b);
    try std.testing.expectEqualSlices(f32, device.tensorValues(tensor_a), &[_]f32{ 3.0, 6.0, 9.0, 12.0 });

    device.tensorOpMultiplication(tensor_a, tensor_b);
    try std.testing.expectEqualSlices(f32, device.tensorValues(tensor_a), &[_]f32{ 6.0, 24.0, 54.0, 96.0 });

    device.tensorOpDivision(tensor_a, tensor_b);
    try std.testing.expectEqualSlices(f32, device.tensorValues(tensor_a), &[_]f32{ 3.0, 6.0, 9.0, 12.0 });

    device.tensorOpExp(tensor_a);
    try std.testing.expectEqualSlices(f32, device.tensorValues(tensor_a), &[_]f32{
        std.math.exp(3.0),
        std.math.exp(6.0),
        std.math.exp(9.0),
        std.math.exp(12.0),
    });
}

test "tensor ensure row-major" {
    var device = TensorDevice.init(std.testing.allocator);
    defer device.deinit();

    var tensor_a = try device.createTensor(tensor.TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 2, 3 },
    });
    device.setTensorData(tensor_a, &[_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 0 }), 1.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 1 }), 2.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 2 }), 3.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 1, 0 }), 4.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 1, 1 }), 5.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 1, 2 }), 6.0);
}

test "tensor ensure index order" {
    var device = TensorDevice.init(std.testing.allocator);
    defer device.deinit();

    var tensor_a = try device.createTensor(tensor.TensorDescriptor{
        .dimensions_sizes = &[_]usize{ 2, 3, 4 },
    });
    device.setTensorData(tensor_a, &[_]f32{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,

        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
    });
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 0, 0 }), 1.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 0, 1 }), 2.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 0, 2 }), 3.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 0, 3 }), 4.0);

    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 1, 0 }), 5.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 0, 2, 0 }), 9.0);

    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 1, 0, 0 }), 13.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 1, 1, 0 }), 17.0);
    try std.testing.expectEqual(device.tensorValue(tensor_a, &.{ 1, 2, 0 }), 21.0);
}
