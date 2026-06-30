# ComputationalBackends.jl

Zero-dependency Julia package providing a unified execution-backend type hierarchy for the
[jbphyswx](https://github.com/jbphyswx) ecosystem.

## Motivation

Multiple packages (`StructureFunctions.jl`, `ScatteringTransforms.jl`,
`CoarseGrainingEnergyFluxes.jl`, `FlowInvariantTransfer.jl`, …) independently defined
near-identical backend type hierarchies. `ComputationalBackends.jl` is the single source of
truth — downstream packages import these types instead of maintaining local copies.

## Exported Types

### Local compute backends

| Type | Description |
|---|---|
| `SerialBackend` | Serial single-threaded CPU (always available) |
| `ThreadedBackend` | Multi-threaded CPU (OhMyThreads extension in consumer) |
| `GPUBackend{B}` | GPU via KernelAbstractions, parameterized on device backend |
| `AutoBackend` | Resolves to best available at runtime |

### Distribution wrappers (parametric over inner local backend)

| Type | Description |
|---|---|
| `DistributedBackend{Inner}` | Multi-process via Distributed.jl |
| `MPIBackend{Inner, C}` | Multi-rank via MPI.jl (not CPU-only) |

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
