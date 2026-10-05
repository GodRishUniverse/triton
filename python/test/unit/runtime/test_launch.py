import gc
import tracemalloc
import pytest
import pathlib
import os
import numpy as np

import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda, is_hip, run_in_process


@pytest.mark.skipif(not is_hip(), reason="AMD pointer range assertion")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32])
def test_pointer_access_range_option_large_allocation(dtype, fresh_knobs):
    fresh_knobs.amd.use_buffer_ops = True
    fresh_knobs.amd.buffer_ops_analyze_small_tensor_range = False
    if torch.cuda.mem_get_info()[0] < 3 * 2**30:
        pytest.skip("requires 3 GiB of free GPU memory")

    @triton.jit
    def gather_scatter(src, dst, indices, SIZE: tl.constexpr):
        lanes = tl.arange(0, SIZE)
        index = tl.load(indices + lanes)
        value = tl.load(src + index)
        tl.store(dst + index, value)

    # Both views start past 2 GiB. Dynamic indices prevent static range analysis
    # from proving the loads/stores safe for buffer addressing on its own.
    size = 256
    nbytes = size * torch.empty((), dtype=dtype).element_size()
    storage = torch.empty(2**31 + 2 * nbytes, dtype=torch.uint8, device="cuda")
    src = storage[2**31:2**31 + nbytes].view(dtype)
    dst = storage[2**31 + nbytes:].view(dtype)
    values = torch.arange(size, device="cuda").to(dtype)
    src.copy_(values)
    indices = torch.arange(size - 1, -1, -1, dtype=torch.int32, device="cuda")

    default = gather_scatter[(1, )](src, dst, indices, size)
    torch.testing.assert_close(dst, values, atol=0, rtol=0)
    assert "global_load" in default.asm["amdgcn"]
    assert "global_store" in default.asm["amdgcn"]

    dst.zero_()
    bounded = gather_scatter[(1, )](src, dst, indices, size, assume_pointer_access_range_32=('src', 'dst'))
    torch.testing.assert_close(dst, values, atol=0, rtol=0)
    assert "tt.pointer_access_range = 32" in bounded.asm["ttir"]
    assert "global_load" not in bounded.asm["amdgcn"]
    assert "global_store" not in bounded.asm["amdgcn"]
    assert "buffer_load" in bounded.asm["amdgcn"]
    assert "buffer_store" in bounded.asm["amdgcn"]
    assert bounded is not default
    assert gather_scatter[(1, )](src, dst, indices, size) is default
    assert gather_scatter[(1, )](src, dst, indices, size, assume_pointer_access_range_32=('src', 'dst')) is bounded


@pytest.mark.skipif(not is_hip(), reason="AMD pointer range assertion")
@pytest.mark.parametrize("names", [("index", ), ("missing", )])
def test_pointer_access_range_option_non_pointer_argument(names):

    @triton.jit
    def kernel(out, index):
        tl.store(out + index, 1)

    out = torch.empty(32, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="scalar source pointer|does not exist"):
        kernel[(1, )](out, 3, assume_pointer_access_range_32=names)


@pytest.mark.skipif(not is_hip(), reason="AMD conservative pointer range default")
def test_pointer_range_default_access_beyond_view(fresh_knobs):
    fresh_knobs.amd.use_buffer_ops = True
    if torch.cuda.mem_get_info()[0] < 3 * 2**30:
        pytest.skip("requires 3 GiB of free GPU memory")

    @triton.jit
    def kernel(src, out, indices):
        lanes = tl.arange(0, 16)
        index = tl.load(indices + lanes)
        tl.store(out + lanes, tl.load(src + index))

    storage = torch.empty(2**31 + 16, dtype=torch.uint8, device="cuda")
    storage[2**31:].fill_(123)
    # Triton receives a pointer, so this is a valid read within its allocation
    # even though the view shape does not describe the addresses being read.
    view = storage[:16]
    indices = torch.arange(2**31, 2**31 + 16, dtype=torch.int64, device="cuda")
    out = torch.empty(16, dtype=torch.uint8, device="cuda")
    compiled = kernel[(1, )](view, out, indices)
    torch.testing.assert_close(out, torch.full_like(out, 123), atol=0, rtol=0)
    assert "global_load" in compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip(), reason="AMD pointer access range assertion")
@pytest.mark.parametrize("small_storage", [False, True])
@pytest.mark.parametrize(
    "mode", ["i64", "i32_chain", "i32_wrap", "select", "if", "loop", "trans", "unknown_if", "masked", "cancel"])
def test_pointer_access_range_preserves_pointer_values(mode, small_storage, fresh_knobs):
    fresh_knobs.amd.use_buffer_ops = True
    fresh_knobs.amd.buffer_ops_analyze_small_tensor_range = False
    if not small_storage and torch.cuda.mem_get_info()[0] < 3 * 2**30:
        pytest.skip("requires 3 GiB of free GPU memory")

    @triton.jit
    def kernel(src, other, offsets, out, MODE: tl.constexpr):
        lanes = tl.arange(0, 4)
        value = tl.load(src + lanes)
        offset = tl.load(offsets + lanes)
        if MODE == "i64":
            pointer = src + offset
        elif MODE == "i32_chain":
            pointer = src + offset.to(tl.int32)
            pointer = pointer + offset.to(tl.int32)
        elif MODE == "i32_wrap":
            wrapped = offset.to(tl.int32) + lanes
            pointer = src + wrapped
        elif MODE == "select":
            pointer = tl.where(tl.load(offsets) > 0, src + offset, other + offset)
        elif MODE == "if":
            if tl.load(offsets) > 0:
                pointer = src + offset
            else:
                pointer = other + offset
        elif MODE == "trans":
            matrix = src + tl.reshape(offset, (2, 2))
            pointer = tl.reshape(tl.trans(matrix), (4, ))
        elif MODE == "unknown_if":
            if tl.load(offsets) > 0:
                pointer = src + offset
            else:
                pointer = (src.to(tl.uint64) + offset.to(tl.uint64)).to(tl.pointer_type(tl.uint8))
        elif MODE == "masked":
            pointer = src + offset
            value = tl.load(pointer, lanes == 0, 77)
        elif MODE == "cancel":
            pointer = src + offset
            pointer = pointer - offset + lanes
        else:
            pointer = src + lanes
            for _ in range(2):
                pointer = tl.where(tl.load(offsets) > 0, pointer, other + lanes)
                pointer = pointer + offset.to(tl.int32)
        delta = pointer.to(tl.uint64) - src.to(tl.uint64)
        tl.store(out + lanes, delta.to(tl.int64))
        tl.store(out + 4 + lanes, value.to(tl.int64))

    # Far pointer values are observed, but the only selected live accesses read
    # bytes 0..3. The small-storage case also carries the ordinary S attribute.
    src = torch.empty(16 if small_storage else 2**31 + 16, device="cuda", dtype=torch.uint8)
    src[:4].copy_(torch.arange(4, device="cuda", dtype=torch.uint8))
    other = torch.empty(16, device="cuda", dtype=torch.uint8)
    if mode in ("i64", "select", "if", "trans", "unknown_if", "masked", "cancel"):
        offsets = torch.arange(2**31, 2**31 + 4, device="cuda", dtype=torch.int64)
        expected = list(range(2**31, 2**31 + 4))
        if mode == "trans":
            expected = [2**31, 2**31 + 2, 2**31 + 1, 2**31 + 3]
        elif mode == "masked":
            offsets[0] = 0
            offsets[1] = -1
            expected[0] = 0
            expected[1] = -1
        elif mode == "cancel":
            expected = list(range(4))
    elif mode == "i32_wrap":
        offsets = torch.full((4, ), 2**31 - 1, device="cuda", dtype=torch.int64)
        expected = [2**31 - 1, -2**31, -2**31 + 1, -2**31 + 2]
    else:
        offsets = torch.full((4, ), 2**30, device="cuda", dtype=torch.int64)
        expected = [2**31 + i if mode == "loop" else 2**31 for i in range(4)]
    out = torch.empty(8, device="cuda", dtype=torch.int64)
    compiled = kernel[(1, )](src, other, offsets, out, mode, assume_pointer_access_range_32=("src", ))
    assert out.cpu().tolist() == expected + ([0, 77, 77, 77] if mode == "masked" else list(range(4)))
    assert "tt.pointer_access_range = 32" in compiled.asm["ttir"]
    assert "buffer_load" in compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip(), reason="AMD pointer access range assertion")
@pytest.mark.parametrize("mode", ["select", "if", "loop"])
@pytest.mark.parametrize("choose_src", [False, True])
def test_pointer_access_range_mixed_origins(mode, choose_src, fresh_knobs):
    fresh_knobs.amd.use_buffer_ops = True
    fresh_knobs.amd.buffer_ops_analyze_small_tensor_range = False

    @triton.jit
    def kernel(src, other, offsets, out, choose, MODE: tl.constexpr):
        lanes = tl.arange(0, 4)
        offset = tl.load(offsets + lanes)
        if MODE == "select":
            pointer = tl.where(choose, src + offset, other + offset)
        elif MODE == "if":
            if choose:
                pointer = src + offset
            else:
                pointer = other + offset
        else:
            pointer = src + lanes
            for _ in range(2):
                pointer = tl.where(choose, pointer, other + offset)
        values = tl.load(pointer)
        delta = pointer.to(tl.uint64) - src.to(tl.uint64)
        tl.store(out + lanes, delta.to(tl.int64))
        tl.store(out + 4 + lanes, values.to(tl.int64))

    src = torch.arange(16, device="cuda", dtype=torch.uint8)
    other = torch.arange(16, device="cuda", dtype=torch.uint8) + 100
    offsets = torch.arange(4, device="cuda", dtype=torch.int64)
    out = torch.empty(8, device="cuda", dtype=torch.int64)
    kernel[(1, )](src, other, offsets, out, choose_src, mode, assume_pointer_access_range_32=("src", ))
    base_delta = 0 if choose_src else other.data_ptr() - src.data_ptr()
    expected = [base_delta + i for i in range(4)] + [i + (0 if choose_src else 100) for i in range(4)]
    assert out.cpu().tolist() == expected


@pytest.mark.skipif(not is_hip(), reason="AMD pointer access range assertion")
def test_pointer_access_range_full_width_boundary(fresh_knobs):
    fresh_knobs.amd.use_buffer_ops = True
    fresh_knobs.amd.buffer_ops_analyze_small_tensor_range = False
    if torch.cuda.mem_get_info()[0] < 3 * 2**30:
        pytest.skip("requires 3 GiB of free GPU memory")

    @triton.jit
    def kernel(src, offsets, out):
        lanes = tl.arange(0, 4)
        offset = tl.load(offsets + lanes)
        tl.store(out + lanes, tl.load(src + offset))

    src = torch.empty(2**29 + 4, device="cuda", dtype=torch.float32)
    # The last live access ends at byte 2**31 - 4, inside the exclusive bound.
    offsets = torch.arange(2**29 - 5, 2**29 - 1, device="cuda", dtype=torch.int64)
    src[offsets] = torch.arange(4, device="cuda", dtype=torch.float32) + 7
    out = torch.empty(4, device="cuda", dtype=torch.float32)
    compiled = kernel[(1, )](src, offsets, out, assume_pointer_access_range_32=("src", ))
    torch.testing.assert_close(out, torch.arange(4, device="cuda", dtype=torch.float32) + 7, atol=0, rtol=0)
    assert "buffer_load" in compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip(), reason="AMD pointer access range assertion")
@pytest.mark.parametrize("kind", ["tuple", "descriptor", "constexpr", "gluon"])
def test_pointer_access_range_rejects_unsupported_arguments(kind):
    from triton.tools.tensor_descriptor import TensorDescriptor
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @triton.jit
    def kernel(arg, out, SIZE: tl.constexpr):
        tl.store(out, SIZE)

    @gluon.jit
    def gluon_kernel(arg, out, SIZE: gl.constexpr):
        gl.store(out, SIZE)

    out = torch.empty(32, device="cuda", dtype=torch.int32)
    arg = out
    names = ("arg", )
    if kind == "tuple":
        arg = (out, out)
    elif kind == "descriptor":
        arg = TensorDescriptor(out, [32], [1], [32])
    elif kind == "constexpr":
        names = ("SIZE", )
    else:
        kernel = gluon_kernel
    with pytest.raises(ValueError, match="scalar source pointer|not Gluon"):
        kernel[(1, )](arg, out, 32, assume_pointer_access_range_32=names)


def test_metadata() -> None:

    used_hook = False

    def _launch_metadata(grid, kernel, args):
        ret = dict()
        ret["grid"] = grid
        ret["value"] = args["x"]
        return ret

    def hook(launch_metadata):
        nonlocal used_hook
        metadata = launch_metadata.get()
        assert metadata["grid"] == (1, 3, 2)
        assert metadata["value"] == 6
        used_hook = True

    @triton.jit(launch_metadata=_launch_metadata)
    def kernel(x):
        pass

    # launch kernel
    triton.knobs.runtime.launch_enter_hook.add(hook)
    kernel[(1, 3, 2)](6)
    triton.knobs.runtime.launch_enter_hook.remove(hook)
    assert used_hook


@pytest.mark.skipif(not is_cuda(), reason="max_occupancy requires CUDA")
@pytest.mark.parametrize("max_occupancy", [None, 1, 2, 4])
@pytest.mark.parametrize("num_ctas", [1, 2])
def test_max_occupancy_launch(max_occupancy, num_ctas, device, monkeypatch, fresh_triton_cache):
    import ctypes
    from triton.backends.nvidia.driver import libcuda_dirs

    if num_ctas > 1 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("CTA clusters require Hopper or newer")

    @triton.jit
    def kernel(out):
        ptr = out + tl.program_id(0)
        tl.store(ptr, tl.load(ptr) + 1)

    out = torch.zeros(16, device=device, dtype=torch.int32)
    compiled = kernel.warmup(out, grid=(16, ), num_warps=4, num_ctas=num_ctas, max_occupancy=max_occupancy)
    assert compiled.metadata.shared == 0
    assert compiled.packed_metadata[2] == 0

    loaded_shared = []
    utils = triton.runtime.driver.active.utils
    original_load = utils.load_binary

    def load_binary(*args):
        loaded_shared.append(args[2])
        return original_load(*args)

    monkeypatch.setattr(utils, "load_binary", load_binary)
    launcher = compiled.run
    assert loaded_shared == [launcher.shared]
    if max_occupancy is None:
        assert launcher.shared == 0
    else:
        properties = utils.get_device_properties(triton.runtime.driver.active.get_current_device())
        assert (max_occupancy + 1) * launcher.shared > properties["max_shared_mem_per_multiprocessor"]
        if num_ctas == 1:
            # Check CUDA's occupancy calculation, independently of the reservation formula.
            cuda = ctypes.CDLL(os.path.join(libcuda_dirs()[0], "libcuda.so.1"))
            occupancy = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
            occupancy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
            occupancy.restype = ctypes.c_int
            blocks = ctypes.c_int()
            assert occupancy(ctypes.byref(blocks), compiled.function, compiled.metadata.num_warps * 32,
                             launcher.shared) == 0
            assert 0 < blocks.value <= max_occupancy

    shared_sizes = []
    original_launch = launcher.launch

    def launch(*args):
        shared_sizes.append(args[7][2])
        return original_launch(*args)

    monkeypatch.setattr(launcher, "launch", launch)
    compiled[(16, 1, 1)](out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled[(16, 1, 1)](out)
    graph.replay()
    graph.replay()
    torch.testing.assert_close(out, torch.full_like(out, 3))
    assert shared_sizes == [launcher.shared, launcher.shared]


@pytest.mark.skipif(not is_cuda(), reason="max_occupancy requires CUDA")
def test_max_occupancy_preserves_required_shared_memory(device):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import async_copy

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Async copy requires Ampere or newer")

    @gluon.jit
    def kernel(inp, out):
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        offsets = gl.arange(0, 16384, layout=layout)
        shared = gl.allocate_shared_memory(gl.float32, [16384], gl.SwizzledSharedLayout(1, 1, 1, [0]))
        async_copy.async_load(shared, inp + offsets)
        async_copy.commit_group()
        async_copy.wait_group(0)
        gl.store(out + offsets, shared.load(layout))

    inp = torch.randn(16384, device=device)
    out = torch.empty_like(inp)
    compiled = kernel[(1, )](inp, out, max_occupancy=4)
    assert compiled.metadata.shared == 65536
    assert compiled.run.shared == compiled.metadata.shared
    torch.testing.assert_close(out, inp)


def _check_memory_leak(device) -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    tracemalloc.start()
    try:
        inp = torch.randn(10, device=device)
        out = torch.randn(10, device=device)
        kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        begin, _ = tracemalloc.get_traced_memory()
        for _ in range(100):
            kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        end, _ = tracemalloc.get_traced_memory()
        assert end - begin < 1000
    finally:
        tracemalloc.stop()


def test_memory_leak(device) -> None:
    # tracemalloc counts allocations from every thread, including xdist's
    # concurrent work-stealing requests. Measure launches in a separate process.
    result = run_in_process(_check_memory_leak, (device, ))
    if result is not None:
        assert result.exc is None, result.exc


def test_load_hook() -> None:

    used_start_hook = False
    start_hash = None

    def hook_start(module, function, name, metadata_group, hash):
        nonlocal used_start_hook
        nonlocal start_hash
        start_hash = hash
        used_start_hook = True

    used_end_hook = False
    end_hash = None

    def hook_end(module, function, name, metadata_group, hash):
        nonlocal used_end_hook
        nonlocal end_hash
        end_hash = hash
        used_end_hook = True

    @triton.jit
    def kernel(x):
        pass

    # launch kernel
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end)
    kernel[(1, 3, 2)](6)
    assert used_start_hook
    assert used_end_hook
    assert start_hash == end_hash
    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end)


def test_multiple_hooks() -> None:

    start0 = False
    end0 = False
    start1 = False
    end1 = False

    def hook_start0(module, function, name, metadata_group, hash):
        nonlocal start0
        start0 = True

    def hook_end0(module, function, name, metadata_group, hash):
        nonlocal end0
        end0 = True

    def hook_start1(module, function, name, metadata_group, hash):
        nonlocal start1
        start1 = True

    def hook_end1(module, function, name, metadata_group, hash):
        nonlocal end1
        end1 = True

    triton.knobs.runtime.kernel_load_start_hook.add(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end1)

    @triton.jit
    def kernel(x):
        pass

    kernel[(1, )](6)

    assert start0
    assert end0
    assert start1
    assert end1

    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end1)


@pytest.mark.parametrize("options", [
    {"num_warps": 1},
    {"enable_fp_fusion": False},
    {"extern_libs": {}},
])
def test_launch_with_options(options, monkeypatch) -> None:
    if "extern_libs" in options:
        # copied from tutorials/07-extern-functions.py
        current_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
        if is_cuda():
            libdir = current_dir.parent.parent.parent.parent / 'third_party/nvidia/backend/lib'
            options["extern_libs"] = {"libdevice": str(libdir / 'libdevice.10.bc')}
        elif is_hip():
            libdir = current_dir.parent.parent.parent.parent / 'third_party/amd/backend/lib'
            options["extern_libs"] = {"ocml": str(libdir / 'ocml.bc'), "ockl": str(libdir / 'ockl.bc')}

    compile_info = {}
    counter = 0

    def compile_info_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):
        nonlocal compile_info
        compile_info = compile

    def cache_hook(*args, **kwargs):
        nonlocal counter
        counter += 1

    @triton.jit
    def kernel(x):
        pass

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", compile_info_hook)
    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", cache_hook)

    # run first without options
    kernel[(1, 1, 1)](6)
    assert counter == 1

    # run with options, should lead to new compilation
    kernel[(1, 1, 1)](6, **options)
    assert counter == 2

    # run a second time for testing kernel-cache look-up
    kernel[(1, 1, 1)](6, **options)
    assert counter == 2

    # check the options are passed on to compile_info correctly
    option_key, option_val = next(iter(options.items()))
    if option_key == "extern_libs":
        # HIPOptions overwrite the extern_libs option, so we skip the test
        # passing and specializing options still is tested
        if not is_hip():
            assert compile_info[option_key] == tuple(option_val.items())
    else:
        assert compile_info[option_key] == option_val


@pytest.mark.interpreter
def test_pre_run_hooks(device):

    @triton.jit
    def add_kernel(a_ptr, n_elements: tl.constexpr):
        offsets = tl.arange(0, n_elements)
        a = tl.load(a_ptr + offsets)
        a += 2
        tl.store(a_ptr + offsets, a)

    def my_hook(*args, **kwargs):
        args[0].zero_()

    add_kernel.add_pre_run_hook(my_hook)

    n_elements = 4
    a = torch.ones(n_elements, device=device, dtype=torch.int32)
    add_kernel[(1, )](a, n_elements)
    assert torch.all(a == 2)

    a = torch.ones(n_elements, device=device, dtype=torch.int32)
    add_kernel.run(a, n_elements, grid=(1, ), warmup=False)
    assert torch.all(a == 2)


def test_interpreter_implicit_cvt_bool() -> None:
    from triton.runtime.interpreter import _implicit_cvt

    value = _implicit_cvt(True)

    assert value.dtype == tl.int1
    assert value.handle.data.dtype == np.bool_
    assert bool(value.handle.data[0]) is True


@pytest.mark.skipif(not is_hip(), reason="requires HIP")
def test_wgp_cu_mode_launch_argument():
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch.startswith(("gfx10", "gfx11", "gfx120")):
        pytest.skip("target has no WGP/CU mode distinction")

    @triton.jit
    def add_one(x_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1)

    x = torch.zeros(64, device="cuda")
    for options, mode in [({}, 1), ({"wgp_cu_mode": "wgp"}, 1), ({"wgp_cu_mode": "cu"}, 0)]:
        kernel = add_one[(1, )](x, BLOCK=64, **options)
        assert f".amdhsa_workgroup_processor_mode {mode}" in kernel.asm["amdgcn"]
    assert torch.equal(x, torch.full_like(x, 3))
