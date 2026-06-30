"""
    ComputationalBackends

Zero-dependency Julia package providing a unified execution-backend type hierarchy for the
jbphyswx ecosystem.  Every package that needs to dispatch on *where/how* computation runs
(serial CPU, multi-threaded, GPU, distributed, MPI) imports these types from here instead of
defining its own copy.

Two orthogonal concerns are encoded in the type tree:

- **Local compute backend** — what one process/rank computes on:
  [`SerialBackend`](@ref), [`ThreadedBackend`](@ref) (OhMyThreads ext),
  [`GPUBackend{B}`](@ref) (KernelAbstractions ext).

- **Distribution wrapper** — how work is split across processes, **parametric over the inner
  local backend**: [`DistributedBackend{Inner}`](@ref) (Distributed ext),
  [`MPIBackend{Inner,C}`](@ref) (MPI ext).  The parametric form makes layouts like
  `DistributedBackend{GPUBackend{CUDABackend}}` (multi-node multi-GPU) and
  `MPIBackend{ThreadedBackend}` (hybrid MPI+threads) expressible.

[`AutoBackend`](@ref) resolves to the best available local backend at runtime.

Heavy backend *implementations* live in consumer-package extensions; this package only
defines the dispatch types and a few pure helpers.
"""
module ComputationalBackends

export AbstractExecutionBackend
export SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend
export local_backend, is_distributed, resolve_backend

# ──────────────────────────────────────────────────────────────────────────────
# Abstract root
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractExecutionBackend

Supertype for all execution backends — local compute backends ([`SerialBackend`](@ref),
[`ThreadedBackend`](@ref), [`GPUBackend`](@ref)) and distribution wrappers
([`DistributedBackend`](@ref), [`MPIBackend`](@ref)).
"""
abstract type AbstractExecutionBackend end

# ──────────────────────────────────────────────────────────────────────────────
# Local compute backends
# ──────────────────────────────────────────────────────────────────────────────

"""
    SerialBackend <: AbstractExecutionBackend

Serial (CPU, single-threaded) execution.  Always available, no extension needed.

This is the reference implementation that all other backends are validated against.

# Examples
```julia
result = compute(args...; backend = SerialBackend())
```
"""
struct SerialBackend <: AbstractExecutionBackend end

"""
    ThreadedBackend <: AbstractExecutionBackend

Multi-threaded CPU execution (typically via OhMyThreads.jl in the consumer package's
extension).

Use when multiple CPU threads are available (`Threads.nthreads() > 1`) and shared-memory
parallelism is suitable.

# Examples
```julia
result = compute(args...; backend = ThreadedBackend())
```
"""
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B} <: AbstractExecutionBackend

GPU-accelerated execution via KernelAbstractions.jl.  Parameterized by the target GPU
backend object `B` (e.g. `CUDA.CUDABackend()`, `KernelAbstractions.CPU()`).

# Examples
```julia
using CUDA
result = compute(args...; backend = GPUBackend(CUDABackend()))

# CPU backend for testing parity:
using KernelAbstractions: CPU
result = compute(args...; backend = GPUBackend(CPU()))
```
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"""
    AutoBackend <: AbstractExecutionBackend

Automatic backend selection based on runtime state.

Default resolution order (via [`resolve_backend`](@ref)):
1. `ThreadedBackend()` when `Threads.nthreads() > 1`
2. `SerialBackend()` otherwise

Consumer packages may override `resolve_backend` to add GPU or distributed detection.

# Examples
```julia
result = compute(args...; backend = AutoBackend())   # chooses automatically
```
"""
struct AutoBackend <: AbstractExecutionBackend end

# ──────────────────────────────────────────────────────────────────────────────
# Distribution wrappers (parametric over inner local backend)
# ──────────────────────────────────────────────────────────────────────────────

"""
    DistributedBackend{Inner <: AbstractExecutionBackend} <: AbstractExecutionBackend
    DistributedBackend(inner = SerialBackend())

Distribute work across worker processes (via Distributed.jl), each running `inner` locally.
Parametric over the inner local backend so that layouts like
`DistributedBackend(ThreadedBackend())` (multithreaded workers) are expressible.

# Examples
```julia
using Distributed; addprocs(4)
result = compute(args...; backend = DistributedBackend())

# hybrid distributed + threaded:
result = compute(args...; backend = DistributedBackend(ThreadedBackend()))
```
"""
struct DistributedBackend{Inner <: AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend{Inner <: AbstractExecutionBackend, C} <: AbstractExecutionBackend
    MPIBackend(inner = SerialBackend(); comm = nothing)

Multi-rank execution via MPI.jl, parametric on the per-rank `inner` backend and communicator
type `C`.  Each rank computes its share with `inner`, then partial results are combined
(e.g. `MPI.Allreduce!`).  `comm = nothing` means the consumer extension uses
`MPI.COMM_WORLD` (the core package cannot reference MPI).

Not CPU-only: `MPIBackend(GPUBackend(CUDABackend()))` targets multi-GPU clusters;
`MPIBackend(ThreadedBackend())` is hybrid MPI+threads.

# Examples
```julia
using MPI; MPI.Init()
result = compute(args...; backend = MPIBackend(ThreadedBackend()))
```
"""
struct MPIBackend{Inner <: AbstractExecutionBackend, C} <: AbstractExecutionBackend
    inner::Inner
    comm::C
end
# `comm = nothing` ⇒ the MPI extension uses `MPI.COMM_WORLD` (core cannot reference MPI).
MPIBackend(inner::AbstractExecutionBackend = SerialBackend(); comm = nothing) =
    MPIBackend(inner, comm)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    local_backend(backend) -> AbstractExecutionBackend

The per-process compute backend: the wrapped `inner` for a distribution wrapper, else the
backend itself.
"""
local_backend(b::AbstractExecutionBackend) = b
local_backend(b::DistributedBackend) = b.inner
local_backend(b::MPIBackend) = b.inner

"""
    is_distributed(backend) -> Bool

`true` if `backend` distributes work across processes (i.e. is a `DistributedBackend` or
`MPIBackend`).
"""
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

"""
    resolve_backend(backend) -> AbstractExecutionBackend

Resolve [`AutoBackend`](@ref) to a concrete local backend instance; all other backends are
returned as-is.

Default resolution: `ThreadedBackend()` if `Threads.nthreads() > 1`, else `SerialBackend()`.
"""
resolve_backend(backend::AbstractExecutionBackend) = backend
resolve_backend(::AutoBackend) = Threads.nthreads() > 1 ? ThreadedBackend() : SerialBackend()

end # module ComputationalBackends
