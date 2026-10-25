"""
    ComputationalBackends

Dispatch tags for where/how computation runs: serial, threaded, GPU, distributed, or MPI.

Hierarchy (subtype at the level you need):

```
AbstractComputationalBackend
└── AbstractExecutionBackend
│   ├── AbstractLocalBackend
│   │   ├── AbstractSerialBackend     ← SerialBackend
│   │   ├── AbstractThreadedBackend   ← ThreadedBackend
│   │   └── AbstractGPUBackend        ← GPUBackend{B}
│   ├── AbstractDistributedBackend    ← DistributedBackend{Inner}
│   ├── AbstractMPIBackend            ← MPIBackend{Inner,C}
│   └── AbstractAutoBackend
└── ...
```

Consumer methods should dispatch on the abstracts (`::AbstractSerialBackend`, …) so user-defined
subtypes participate.  The concrete types above are the defaults.

[`AutoBackend`](@ref) is not local, so it cannot be a distribution `Inner`. What Auto resolves to is
each consumer's policy: [`resolve_backend`](@ref)`(::AutoBackend)` throws, a consumer resolves Auto
with a function of its own, and [`default_resolve_auto_backend`](@ref) is an opt-in convenience.

This package defines tags and helpers only.  Kernel implementations live in consumer extensions.
"""
module ComputationalBackends
export AbstractComputationalBackend
export AbstractExecutionBackend, AbstractLocalBackend
export AbstractSerialBackend, AbstractThreadedBackend, AbstractGPUBackend, AbstractDistributedBackend, AbstractMPIBackend, AbstractAutoBackend
export SerialBackend, ThreadedBackend, GPUBackend, AutoBackend, DistributedBackend, MPIBackend

# ──────────────────────────────────────────────────────────────────────────────
# Abstract roots
# ──────────────────────────────────────────────────────────────────────────────
abstract type AbstractComputationalBackend end

"""
    AbstractExecutionBackend

Supertype for all execution backends.
"""
abstract type AbstractExecutionBackend <: AbstractComputationalBackend end

"""
    AbstractLocalBackend <: AbstractExecutionBackend

Per-process compute backends.  Distribution wrappers require `Inner <: AbstractLocalBackend`.
"""
abstract type AbstractLocalBackend <: AbstractExecutionBackend end

"""
    AbstractSerialBackend <: AbstractLocalBackend

Serial CPU execution.  Subtype to carry policy (cache blocking, problem-size cutoffs, …).
Default instance: [`SerialBackend`](@ref).
"""
abstract type AbstractSerialBackend <: AbstractLocalBackend end

"""
    AbstractThreadedBackend <: AbstractLocalBackend

Shared-memory multithreaded execution.  Subtype for scheduler / `ntasks` / threadpool variants.
Default instance: [`ThreadedBackend`](@ref).
"""
abstract type AbstractThreadedBackend <: AbstractLocalBackend end

"""
    AbstractGPUBackend <: AbstractLocalBackend

Device execution.  Subtype for vendor- or stream-specific wrappers.
Default instance: [`GPUBackend`](@ref).
"""
abstract type AbstractGPUBackend <: AbstractLocalBackend end

"""
    AbstractDistributedBackend <: AbstractExecutionBackend

Multi-process distribution.  Default instance: [`DistributedBackend`](@ref).
Custom subtypes must define [`local_backend`](@ref) if they are not `DistributedBackend`.
"""
abstract type AbstractDistributedBackend <: AbstractExecutionBackend end

"""
    AbstractMPIBackend <: AbstractExecutionBackend

MPI multi-rank distribution.  Default instance: [`MPIBackend`](@ref).
Custom subtypes must define [`local_backend`](@ref) if they are not `MPIBackend`.
"""
abstract type AbstractMPIBackend <: AbstractExecutionBackend end

"""
    AbstractAutoBackend <: AbstractExecutionBackend

Unresolved execution backend, resolved by each consumer's own policy.
Default instance: [`AutoBackend`](@ref).
"""
abstract type AbstractAutoBackend <: AbstractExecutionBackend end

# ──────────────────────────────────────────────────────────────────────────────
# Default concrete local backends
# ──────────────────────────────────────────────────────────────────────────────

"""
    SerialBackend <: AbstractSerialBackend

Default serial CPU backend.  Always available.
"""
struct SerialBackend <: AbstractSerialBackend end

"""
    ThreadedBackend <: AbstractThreadedBackend

Default multithreaded CPU backend (typically OhMyThreads in a consumer extension).
"""
struct ThreadedBackend <: AbstractThreadedBackend end

"""
    GPUBackend{B} <: AbstractGPUBackend

Default GPU backend via KernelAbstractions, parameterized by device backend object `B`
(e.g. `CUDA.CUDABackend()`, `KernelAbstractions.CPU()`).
"""
struct GPUBackend{B} <: AbstractGPUBackend
    backend::B
end

"""
    AutoBackend <: AbstractAutoBackend

Entry-point selector, resolved by each consumer. It is not local, so it cannot be a distribution
`Inner`.
"""
struct AutoBackend <: AbstractAutoBackend end

# ──────────────────────────────────────────────────────────────────────────────
# Default concrete distribution wrappers
# ──────────────────────────────────────────────────────────────────────────────

"""
    DistributedBackend{Inner <: AbstractLocalBackend} <: AbstractDistributedBackend
    DistributedBackend(inner = SerialBackend())

Distributed.jl workers, each running `inner` locally.
"""
struct DistributedBackend{Inner <: AbstractLocalBackend} <: AbstractDistributedBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend{Inner <: AbstractLocalBackend, C} <: AbstractMPIBackend
    MPIBackend(inner = SerialBackend(); comm = nothing)

MPI ranks, each running `inner` locally.  `comm = nothing` leaves the communicator to the consumer
(conventionally `MPI.COMM_WORLD`).
"""
struct MPIBackend{Inner <: AbstractLocalBackend, C} <: AbstractMPIBackend
    inner::Inner
    comm::C
end
MPIBackend(inner::AbstractLocalBackend = SerialBackend(); comm = nothing) =
    MPIBackend(inner, comm)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    is_local_backend(backend) -> Bool

`true` for any [`AbstractLocalBackend`](@ref).
"""
is_local_backend(::AbstractExecutionBackend) = false
is_local_backend(::AbstractLocalBackend) = true

"""
    local_backend(backend) -> AbstractLocalBackend

Per-process compute backend.  Unwraps [`DistributedBackend`](@ref) / [`MPIBackend`](@ref)
recursively.  Custom [`AbstractDistributedBackend`](@ref) / [`AbstractMPIBackend`](@ref) subtypes
should define their own method.
"""
local_backend(b::AbstractLocalBackend) = b
local_backend(b::DistributedBackend) = local_backend(b.inner)
local_backend(b::MPIBackend) = local_backend(b.inner)
function local_backend(b::AbstractExecutionBackend)
    throw(ArgumentError(
        "local_backend expects a local backend or DistributedBackend/MPIBackend; got $(typeof(b)). " *
        "Custom distribution subtypes must define local_backend themselves. " *
        "Resolve AutoBackend with resolve_backend first.",
    ))
end

"""
    is_distributed(backend) -> Bool

`true` for [`AbstractDistributedBackend`](@ref) or [`AbstractMPIBackend`](@ref).
"""
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::AbstractDistributedBackend) = true
is_distributed(::AbstractMPIBackend) = true

"""
    resolve_backend(backend) -> AbstractExecutionBackend

`backend` unchanged. Throws for an [`AbstractAutoBackend`](@ref). Consumers resolve Auto with a
function they own, or call [`default_resolve_auto_backend`](@ref); they add no methods to this one.
"""
resolve_backend(backend::AbstractExecutionBackend) = backend
resolve_backend(::AbstractAutoBackend) = throw(ArgumentError(
    "resolve_backend(::AutoBackend) has no method; resolve AutoBackend with the consuming " *
    "package's own function, or call ComputationalBackends.default_resolve_auto_backend.",
))

"""
    default_resolve_auto_backend(backend::AbstractAutoBackend) -> Union{SerialBackend,ThreadedBackend}

[`ThreadedBackend`](@ref) when `Threads.nthreads() > 1`, else [`SerialBackend`](@ref). An opt-in
convenience for a consumer whose Auto policy is exactly this.
"""
default_resolve_auto_backend(::AbstractAutoBackend) =
    Threads.nthreads() > 1 ? ThreadedBackend() : SerialBackend()

end # module ComputationalBackends
