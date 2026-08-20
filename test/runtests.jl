using ComputationalBackends: ComputationalBackends
using Test: Test

const CB = ComputationalBackends

Test.@testset "ComputationalBackends.jl" begin

    Test.@testset "Construction" begin
        Test.@test CB.SerialBackend() isa CB.AbstractSerialBackend
        Test.@test CB.ThreadedBackend() isa CB.AbstractThreadedBackend
        Test.@test CB.GPUBackend(:mock_device) isa CB.AbstractGPUBackend
        Test.@test CB.AutoBackend() isa CB.AbstractExecutionBackend
        Test.@test !(CB.AutoBackend() isa CB.AbstractLocalBackend)
        Test.@test CB.DistributedBackend() isa CB.AbstractDistributedBackend
        Test.@test CB.MPIBackend() isa CB.AbstractMPIBackend
    end

    Test.@testset "Subtyping" begin
        Test.@test CB.AbstractSerialBackend <: CB.AbstractLocalBackend
        Test.@test CB.AbstractThreadedBackend <: CB.AbstractLocalBackend
        Test.@test CB.AbstractGPUBackend <: CB.AbstractLocalBackend
        Test.@test CB.SerialBackend <: CB.AbstractSerialBackend
        Test.@test CB.ThreadedBackend <: CB.AbstractThreadedBackend
        Test.@test CB.GPUBackend <: CB.AbstractGPUBackend
        Test.@test CB.AbstractLocalBackend <: CB.AbstractExecutionBackend
        Test.@test CB.AbstractDistributedBackend <: CB.AbstractExecutionBackend
        Test.@test CB.AbstractMPIBackend <: CB.AbstractExecutionBackend
        Test.@test CB.DistributedBackend <: CB.AbstractDistributedBackend
        Test.@test CB.MPIBackend <: CB.AbstractMPIBackend
    end

    Test.@testset "User subtypes participate" begin
        struct BlockedSerial <: CB.AbstractSerialBackend
            block::Int
        end
        struct PoolThreaded <: CB.AbstractThreadedBackend end
        struct MyGPU <: CB.AbstractGPUBackend end

        Test.@test BlockedSerial(64) isa CB.AbstractLocalBackend
        Test.@test CB.is_local_backend(BlockedSerial(64))
        Test.@test CB.local_backend(BlockedSerial(64)) isa BlockedSerial
        Test.@test CB.DistributedBackend(BlockedSerial(32)).inner.block == 32
        Test.@test CB.MPIBackend(PoolThreaded()).inner isa PoolThreaded
        Test.@test MyGPU() isa CB.AbstractGPUBackend
        Test.@test CB.is_local_backend(MyGPU())
    end

    Test.@testset "Inner must be local" begin
        Test.@test_throws MethodError CB.DistributedBackend(CB.AutoBackend())
        Test.@test_throws MethodError CB.MPIBackend(CB.AutoBackend())
        Test.@test_throws MethodError CB.DistributedBackend(CB.MPIBackend())
        Test.@test_throws MethodError CB.MPIBackend(CB.DistributedBackend())
    end

    Test.@testset "GPUBackend parametric" begin
        g = CB.GPUBackend(:test_device)
        Test.@test g.backend === :test_device
        Test.@test CB.GPUBackend{Symbol} === typeof(g)
    end

    Test.@testset "DistributedBackend parametric" begin
        d = CB.DistributedBackend()
        Test.@test d.inner isa CB.SerialBackend
        dt = CB.DistributedBackend(CB.ThreadedBackend())
        Test.@test dt.inner isa CB.ThreadedBackend
        dg = CB.DistributedBackend(CB.GPUBackend(:cuda))
        Test.@test dg.inner isa CB.GPUBackend
        Test.@test dg.inner.backend === :cuda
    end

    Test.@testset "MPIBackend parametric" begin
        m = CB.MPIBackend()
        Test.@test m.inner isa CB.SerialBackend
        Test.@test m.comm === nothing
        mt = CB.MPIBackend(CB.ThreadedBackend())
        Test.@test mt.inner isa CB.ThreadedBackend
        mc = CB.MPIBackend(CB.SerialBackend(); comm = :mock_comm)
        Test.@test mc.comm === :mock_comm
    end

    Test.@testset "local_backend" begin
        Test.@test CB.local_backend(CB.SerialBackend()) isa CB.SerialBackend
        Test.@test CB.local_backend(CB.ThreadedBackend()) isa CB.ThreadedBackend
        Test.@test CB.local_backend(CB.GPUBackend(:x)) isa CB.GPUBackend
        Test.@test CB.local_backend(CB.DistributedBackend()) isa CB.SerialBackend
        Test.@test CB.local_backend(CB.DistributedBackend(CB.ThreadedBackend())) isa CB.ThreadedBackend
        Test.@test CB.local_backend(CB.MPIBackend()) isa CB.SerialBackend
        Test.@test CB.local_backend(CB.MPIBackend(CB.GPUBackend(:g))) isa CB.GPUBackend
        Test.@test_throws ArgumentError CB.local_backend(CB.AutoBackend())
    end

    Test.@testset "is_distributed / is_local_backend" begin
        Test.@test CB.is_local_backend(CB.SerialBackend())
        Test.@test CB.is_local_backend(CB.ThreadedBackend())
        Test.@test CB.is_local_backend(CB.GPUBackend(:x))
        Test.@test !CB.is_local_backend(CB.AutoBackend())
        Test.@test !CB.is_local_backend(CB.DistributedBackend())
        Test.@test !CB.is_local_backend(CB.MPIBackend())

        Test.@test !CB.is_distributed(CB.SerialBackend())
        Test.@test !CB.is_distributed(CB.AutoBackend())
        Test.@test CB.is_distributed(CB.DistributedBackend())
        Test.@test CB.is_distributed(CB.MPIBackend())
    end

    Test.@testset "resolve_backend" begin
        Test.@test CB.resolve_backend(CB.SerialBackend()) isa CB.SerialBackend
        Test.@test CB.resolve_backend(CB.ThreadedBackend()) isa CB.ThreadedBackend
        Test.@test CB.resolve_backend(CB.GPUBackend(:x)) isa CB.GPUBackend
        Test.@test CB.resolve_backend(CB.DistributedBackend()) isa CB.DistributedBackend
        auto = CB.resolve_backend(CB.AutoBackend())
        Test.@test auto isa Union{CB.SerialBackend, CB.ThreadedBackend}
        Test.@test auto === (Threads.nthreads() > 1 ? CB.ThreadedBackend() : CB.SerialBackend())
        Test.@inferred CB.resolve_backend(CB.SerialBackend())
        Test.@inferred CB.resolve_backend(CB.ThreadedBackend())
    end

    Test.@testset "is_gpu_array default" begin
        Test.@test CB.is_gpu_array(rand(3)) === false
        Test.@test CB.is_gpu_array(1) === false
    end

end
