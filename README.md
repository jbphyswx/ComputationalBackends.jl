# ComputationalBackends.jl

Zero-dependency dispatch types for where/how computation runs (serial, threaded, GPU, distributed, MPI).

Dispatch on the abstracts so user subtypes work; use the concrete defaults as instances.

| Abstract | Default concrete |
|----------|------------------|
| `AbstractSerialBackend` | `SerialBackend` |
| `AbstractThreadedBackend` | `ThreadedBackend` |
| `AbstractGPUBackend` | `GPUBackend{B}` |
| `AbstractDistributedBackend` | `DistributedBackend{Inner}` |
| `AbstractMPIBackend` | `MPIBackend{Inner,C}` |

`Inner` on distributors must be `<: AbstractLocalBackend`. `AutoBackend` is not local.

## Helpers

| Function | Behavior |
|----------|----------|
| `local_backend(b)` | Unwrap distributors → local backend |
| `is_distributed(b)` | `true` for distributed/MPI abstracts |
| `is_local_backend(b)` | `true` for local abstracts |
| `resolve_backend(b)` | Identity; `AutoBackend` → `SerialBackend()` |
| `recommend_backend(; threaded)` | Interactive Threaded-vs-Serial (not for hot paths) |
| `is_gpu_array(x)` | Default `false`; extend for device arrays |

## Usage

```julia
using ComputationalBackends: ComputationalBackends as CB

# Library methods dispatch on abstracts:
#   compute!(::CB.AbstractSerialBackend, ...)
#   compute!(::CB.AbstractThreadedBackend, ...)

compute(args...; backend = CB.SerialBackend())
compute(args...; backend = CB.MPIBackend(CB.GPUBackend(cuda_backend)))

# User subtype (e.g. serial policy by problem size):
struct BlockedSerial <: CB.AbstractSerialBackend
    block::Int
end
```
