const std = @import("std");
pub const tensor = @import("tensor.zig");
pub const dag = @import("dag.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
}
