const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Mock CUPTI library
    const mock_cupti = b.addSharedLibrary(.{
        .name = "cupti",
        .target = target,
        .optimize = optimize,
    });
    mock_cupti.addCSourceFile(.{
        .file = b.path("test/mock_cupti.c"),
        .flags = &[_][]const u8{
            "-Wall",
            "-Wextra",
        },
    });
    mock_cupti.linkLibC();

    // Add CUDA include paths
    mock_cupti.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
    mock_cupti.addIncludePath(.{ .cwd_relative = "/opt/cuda/include" });

    b.installArtifact(mock_cupti);

    // Create versioned symlink so libparcagpucupti.so can find it
    const symlink_step = b.addSystemCommand(&[_][]const u8{
        "ln", "-sf", "libcupti.so", "libcupti.so.12"
    });
    symlink_step.setCwd(.{ .cwd_relative = b.getInstallPath(.lib, "") });
    symlink_step.step.dependOn(&mock_cupti.step);
    b.getInstallStep().dependOn(&symlink_step.step);

    // Note: libparcagpucupti.so should be built with CMake for production use.
    // This Zig build only builds the test infrastructure (mock CUPTI + test program).
    // The test will use the CMake-built library from cupti/build/libparcagpucupti.so

    // Test executable
    const test_exe = b.addExecutable(.{
        .name = "test_cupti_prof",
        .target = target,
        .optimize = optimize,
    });
    test_exe.addCSourceFile(.{
        .file = b.path("test/test_cupti_prof.c"),
        .flags = &[_][]const u8{
            "-Wall",
            "-Wextra",
            "-D_POSIX_C_SOURCE=199309L",
        },
    });
    test_exe.linkLibC();
    test_exe.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
    test_exe.addIncludePath(.{ .cwd_relative = "/opt/cuda/include" });

    // Link against mock CUPTI for headers
    test_exe.linkLibrary(mock_cupti);

    // Add dl library for dynamic loading
    test_exe.linkSystemLibrary("dl");

    // Set rpath so the test can find the mock CUPTI library
    test_exe.addRPath(.{ .cwd_relative = b.getInstallPath(.lib, "") });

    b.installArtifact(test_exe);

    // Run step
    const run_cmd = b.addRunArtifact(test_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // Set LD_LIBRARY_PATH so dlsym can find symbols
    run_cmd.setEnvironmentVariable("LD_LIBRARY_PATH", b.getInstallPath(.lib, ""));

    // Pass library path to test executable
    const lib_path = b.fmt("{s}/libparcagpucupti.so", .{b.getInstallPath(.lib, "")});
    run_cmd.addArg(lib_path);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the test program");
    run_step.dependOn(&run_cmd.step);
}
