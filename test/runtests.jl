using ComputationalBackends: ComputationalBackends
using Test: Test

const CB = ComputationalBackends

Test.@testset "ComputationalBackends.jl" begin

    # ── Construction ─────────────────────────────────────────────────────
    Test.@testset "Construction" begin
        Test.@test CB.SerialBackend() isa CB.AbstractExecutionBackend
        Test.@test CB.ThreadedBackend() isa CB.AbstractExecutionBackend
        Test.@test CB.GPUBackend(:mock_device) isa CB.AbstractExecutionBackend
        Test.@test CB.AutoBackend() isa CB.AbstractExecutionBackend
        Test.@test CB.DistributedBackend() isa CB.AbstractExecutionBackend
        Test.@test CB.MPIBackend() isa CB.AbstractExecutionBackend
    end

    # ── Subtyping ────────────────────────────────────────────────────────
    Test.@testset "Subtyping" begin
        Test.@test CB.SerialBackend <: CB.AbstractExecutionBackend
        Test.@test CB.ThreadedBackend <: CB.AbstractExecutionBackend
        Test.@test CB.GPUBackend <: CB.AbstractExecutionBackend
        Test.@test CB.AutoBackend <: CB.AbstractExecutionBackend
        Test.@test CB.DistributedBackend <: CB.AbstractExecutionBackend
        Test.@test CB.MPIBackend <: CB.AbstractExecutionBackend
    end

    # ── GPUBackend parametric ────────────────────────────────────────────
    Test.@testset "GPUBackend parametric" begin
        g = CB.GPUBackend(:test_device)
        Test.@test g.backend === :test_device
        Test.@test CB.GPUBackend{Symbol} === typeof(g)
    end

    # ── DistributedBackend parametric ────────────────────────────────────
    Test.@testset "DistributedBackend parametric" begin
        # Default inner is SerialBackend
        d = CB.DistributedBackend()
        Test.@test d.inner isa CB.SerialBackend

        # Explicit inner
        dt = CB.DistributedBackend(CB.ThreadedBackend())
        Test.@test dt.inner isa CB.ThreadedBackend

        # Nested composition
        dg = CB.DistributedBackend(CB.GPUBackend(:cuda))
        Test.@test dg.inner isa CB.GPUBackend
        Test.@test dg.inner.backend === :cuda
    end

    # ── MPIBackend parametric ────────────────────────────────────────────
    Test.@testset "MPIBackend parametric" begin
        # Default: SerialBackend inner, nothing comm
        m = CB.MPIBackend()
        Test.@test m.inner isa CB.SerialBackend
        Test.@test m.comm === nothing

        # Explicit inner
        mt = CB.MPIBackend(CB.ThreadedBackend())
        Test.@test mt.inner isa CB.ThreadedBackend
        Test.@test mt.comm === nothing

        # Explicit comm
        mc = CB.MPIBackend(CB.SerialBackend(); comm = :mock_comm)
        Test.@test mc.comm === :mock_comm
    end

    # ── local_backend ────────────────────────────────────────────────────
    Test.@testset "local_backend" begin
        Test.@test CB.local_backend(CB.SerialBackend()) isa CB.SerialBackend
        Test.@test CB.local_backend(CB.ThreadedBackend()) isa CB.ThreadedBackend
        Test.@test CB.local_backend(CB.GPUBackend(:x)) isa CB.GPUBackend

        # Unwraps distribution wrappers
        Test.@test CB.local_backend(CB.DistributedBackend()) isa CB.SerialBackend
        Test.@test CB.local_backend(CB.DistributedBackend(CB.ThreadedBackend())) isa CB.ThreadedBackend
        Test.@test CB.local_backend(CB.MPIBackend()) isa CB.SerialBackend
        Test.@test CB.local_backend(CB.MPIBackend(CB.GPUBackend(:g))) isa CB.GPUBackend
    end

    # ── is_distributed ───────────────────────────────────────────────────
    Test.@testset "is_distributed" begin
        Test.@test !CB.is_distributed(CB.SerialBackend())
        Test.@test !CB.is_distributed(CB.ThreadedBackend())
        Test.@test !CB.is_distributed(CB.GPUBackend(:x))
        Test.@test !CB.is_distributed(CB.AutoBackend())

        Test.@test CB.is_distributed(CB.DistributedBackend())
        Test.@test CB.is_distributed(CB.MPIBackend())
    end

    # ── resolve_backend ──────────────────────────────────────────────────
    Test.@testset "resolve_backend" begin
        # Non-Auto backends pass through
        Test.@test CB.resolve_backend(CB.SerialBackend()) isa CB.SerialBackend
        Test.@test CB.resolve_backend(CB.ThreadedBackend()) isa CB.ThreadedBackend
        Test.@test CB.resolve_backend(CB.GPUBackend(:x)) isa CB.GPUBackend

        # AutoBackend resolves to something concrete
        resolved = CB.resolve_backend(CB.AutoBackend())
        Test.@test resolved isa CB.AbstractExecutionBackend
        Test.@test !(resolved isa CB.AutoBackend)
        # Result depends on Threads.nthreads()
        if Threads.nthreads() > 1
            Test.@test resolved isa CB.ThreadedBackend
        else
            Test.@test resolved isa CB.SerialBackend
        end
    end

end
