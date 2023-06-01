#!/usr/bin/env julia

#Start Test Script
using SeqData
using Test

# Run tests
println("Test SeqData")
@time @test include("seqdata_test.jl")
