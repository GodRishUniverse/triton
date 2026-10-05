triton
======

.. currentmodule:: triton

.. autosummary::
    :toctree: generated
    :nosignatures:

    jit
    autotune
    heuristics
    Config

AMD pointer access range assertions
-----------------------------------

The experimental AMD compilation option ``assume_pointer_access_range_32``
identifies scalar source pointer arguments whose live memory accesses stay
within ``[0, 2**31 - 1)`` bytes of the argument's data pointer. It allows AMD
buffer addressing for small regions of a large allocation when the compiler
cannot prove the offset range:

.. code-block:: python

    kernel[grid](src, dst, indices,
                 assume_pointer_access_range_32=('src', 'dst'))

Names refer to the kernel's source arguments, so constexpr arguments do not
shift their meaning. Unknown names and names selecting integers, constexprs,
tuples or descriptors are rejected. This option is supported for Triton kernels
only; Gluon kernels are rejected. Duplicate names and name order are normalized,
and the option participates in the compilation cache key.

The caller must establish the range for every live access through the selected
pointers, including accesses in called functions. This includes the complete
byte extent of each access, rather than just its starting offset. Masked-out
accesses do not need to satisfy the range. An incorrect assertion can silently
truncate offsets and produce incorrect results. The option does not perform a
runtime check or impose memory bounds. Tensor shape and strides alone do not
establish this property because Triton kernels can address beyond a view while
remaining inside its backing allocation.

The assertion describes memory accesses only. Other pointer uses, including
conversion to integers and pointer comparisons, retain their original 64-bit
semantics. Each original offset expression is evaluated in its original integer
width before sign extension; chained pointer additions accumulate in 64 bits.
The compiler trusts the assertion for buffer conversion only when it retains
that original argument as the base. Unsupported pointer forms conservatively
keep ordinary addressing.

By default, the AMD backend specializes pointers using their ``ptr_range()``
hook, when present, or their PyTorch backing storage size. This behavior is
unchanged when the option is omitted. ``AMDGCN_USE_BUFFER_OPS`` still controls
buffer conversion, and ``AMDGCN_ANALYZE_SMALL_TENSOR_RANGE`` still requests
compiler range analysis even for pointers carrying this assertion.
