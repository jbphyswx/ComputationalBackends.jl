# ComputationalBackends.jl

Zero-dependency Julia package defining dispatch types for execution-backend selection (parallelism, device, distribution).

## Exported Types

### Local compute backends

| Type | Description |
|---|---|
| `SerialBackend` | Serial single-threaded CPU (always available) |
| `ThreadedBackend` | Multi-threaded CPU (e.g. via OhMyThreads.jl) |
| `GPUBackend{B}` | GPU via KernelAbstractions, parameterized on device backend |
| `AutoBackend` | Resolves to best available at runtime |

### Distribution wrappers (parametric over inner local backend)

| Type | Description |
|---|---|
| `DistributedBackend{Inner}` | Multi-process via Distributed.jl |
| `MPIBackend{Inner, C}` | Multi-rank via MPI.jl (not CPU-only) |

All types are subtypes of `AbstractExecutionBackend`.

### Helpers

| Function | Description |
|---|---|
| `local_backend(b)` | Unwrap distribution wrapper → inner backend |
| `is_distributed(b)` | `true` for `DistributedBackend` / `MPIBackend` |
| `resolve_backend(b)` | Resolve `AutoBackend` → concrete backend |

## Usage

```julia
using ComputationalBackends: ComputationalBackends as CB

# Direct use
backend = CB.SerialBackend()

# Auto-resolution
backend = CB.resolve_backend(CB.AutoBackend())

# Composable wrappers
backend = CB.DistributedBackend(CB.ThreadedBackend())   # multithreaded workers
backend = CB.MPIBackend(CB.GPUBackend(cuda_backend))    # multi-GPU cluster
```
