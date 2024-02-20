Threads.nthreads()
begin
    using LinearAlgebra 
    using FFTW
    using FileIO
    using JLD2
    using DiffEqFlux
    using OrdinaryDiffEq
    using BlockArrays
    using LaTeXStrings
    using SparseArrays
    using BSON
    using Distances
    using DifferentialEquations
    using Optimization
    using OptimizationPolyalgorithms
    using Zygote
    using OptimizationOptimJL
    using OptimizationOptimisers
    using DiffEqSensitivity
    using DelimitedFiles
end 


function train_loop(uinit,adtype,optf,train_maxiters,learning_rate,optimizer_type)
    LOSS  = []                              # Loss accumulator
    PRED  = []                              # prediction accumulator
    PARS  = []                              # parameters accumulator
    global iters = 0 ; 
    println("Max iters:", train_maxiters)
    callback = function (θ,l,pred)
        global iters += 1 
        println("Iteration: $iters || Loss: $l")
		flush(stdout)
        append!(PRED, [pred])
        append!(LOSS, l)
        append!(PARS, [θ])     
        false
    end 
    
    if optimizer_type == "Nesterov"
        println("Choosing Nesterov Optimizer.")
        function train_Nesterov(uinit)
            path_checkpoint_new = "./restartCkpt_Nesterov_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);
            # Training
            res = Optimization.solve(optprob,
                                     OptimizationOptimisers.Nesterov(learning_rate, momentum),
                                     callback = callback,
                                     maxiters = train_maxiters)
            println("saving Nesterov checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_Nesterov(uinit)
        jldsave("ptrained_Nesterov.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit

    elseif optimizer_type == "Momentum"
        println("Choosing Momentum Optimizer.")
        function train_Momentum(uinit)
            path_checkpoint_new = "./restartCkpt_Momentum_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);
            # Training
            res = Optimization.solve(optprob,
                                     OptimizationOptimisers.Momentum(learning_rate, momentum),
                                     callback = callback,
                                     maxiters = train_maxiters)
            println("saving Momentum checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_Momentum(uinit)
        jldsave("ptrained_Momentum.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit

    elseif optimizer_type == "GradientDescent"
        println("Choosing Gradient Descent Optimizer.")
        function train_GradientDescent(uinit)
            path_checkpoint_new = "./restartCkpt_GradientDescent_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);
            # Training
            res = Optimization.solve(optprob,
                                     OptimizationOptimisers.Descent(learning_rate),
                                     callback = callback,
                                     maxiters = train_maxiters)
            println("saving Gradient Descent checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_GradientDescent(uinit)
        jldsave("ptrained_GradientDescent.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit

    elseif optimizer_type == "ADAM"
        println("Choosing ADAM Optimizer.")
        function train_ADAM(uinit)
            path_checkpoint_new = "./restartCkpt_ADAM_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);    
            # Training 
            res = Optimization.solve(optprob, 
                                    OptimizationOptimisers.Adam(learning_rate), 
                                    callback = callback, 
                                    maxiters = train_maxiters)
            println("saving ADAM checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_ADAM(uinit)
        jldsave("ptrained_ADAM.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit
        
    elseif optimizer_type == "RAdam"
        println("Choosing RAdam Optimizer.")
        function train_RAdam(uinit)
            path_checkpoint_new = "./restartCkpt_RAdam_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);    
            # Training 
            res = Optimization.solve(optprob, 
                                    OptimizationOptimisers.RAdam(learning_rate), 
                                    callback = callback, 
                                    maxiters = train_maxiters)
            println("saving RAdam checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_RAdam(uinit)
        jldsave("ptrained_RAdam.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit
        
    elseif optimizer_type == "NAdam"
        println("Choosing NAdam Optimizer.")
        function train_NAdam(uinit)
            path_checkpoint_new = "./restartCkpt_NAdam_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);    
            # Training 
            res = Optimization.solve(optprob, 
                                    OptimizationOptimisers.NAdam(learning_rate), 
                                    callback = callback, 
                                    maxiters = train_maxiters)
            println("saving NAdam checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_NAdam(uinit)
        jldsave("ptrained_NAdam.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit

    elseif optimizer_type == "AdamW"
        println("Choosing AdamW Optimizer.")
        function train_AdamW(uinit)
            path_checkpoint_new = "./restartCkpt_AdamW_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);    
            # Training 
            res = Optimization.solve(optprob, 
                                    OptimizationOptimisers.AdamW(learning_rate), 
                                    callback = callback, 
                                    maxiters = train_maxiters)
            println("saving AdamW checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_AdamW(uinit)
        jldsave("ptrained_AdamW.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit

    elseif optimizer_type == "ADADelta"
        println("Choosing ADADelta Optimizer.")
        function train_ADADelta(uinit)
            path_checkpoint_new = "./restartCkpt_ADADelta_new.jld2"
            optprob = Optimization.OptimizationProblem(optf, uinit)
            # Training
            res = Optimization.solve(
                optprob,
                OptimizationOptimisers.ADADelta(),
                callback = callback,
                maxiters = train_maxiters
            )
            println("saving ADADelta checkpoint...")
            jldsave(path_checkpoint_new, ckpt = res)
            return res
        end

        resnew = train_ADADelta(uinit)
        jldsave("ptrained_Adadelta.jld2", p = resnew.u)
        uinit = resnew.u
        return uinit

    elseif optimizer_type == "ADAGrad"
        println("Choosing ADAGrad Optimizer.")
        function train_ADAGrad(uinit)
            path_checkpoint_new = "./restartCkpt_ADAGrad_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);    
            # Training 
            res = Optimization.solve(optprob, 
                                    OptimizationOptimisers.ADAGrad(learning_rate), 
                                    callback = callback, 
                                    maxiters = train_maxiters)
            println("saving ADAGrad checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end

        resnew = train_ADAGrad(uinit)
        jldsave("ptrained_ADAGrad.jld2",p=resnew.u)
        uinit = resnew.u
        return uinit


    elseif optimizer_type == "BFGS"
        println("Choosing BFGS Optimizer.")      
        function train_BFGS(uinit)
            path_checkpoint_new = "./restartCkpt_BFGS_new.jld2"
            optprob = Optimization.OptimizationProblem(optf,uinit);    
            # Training 
            res = Optimization.solve(optprob, 
                                    Optim.BFGS(initial_stepnorm=0.01),
                                    callback = callback,       
                                    allow_f_increases = false,
									maxiters = train_maxiters,
									f_tol = 1e-9);
            println("saving BFGS checkpoint...")
            jldsave(path_checkpoint_new, ckpt=res)
            return res
        end
        
        resnew = train_BFGS(uinit)
        jldsave("ptrained_BFGS.jld2",p=resnew.u)
		println("saved trained params to ptrained_BFGS.jld2")
		uinit = resnew.u
		
        return uinit
        
     else 
        println("Abort, Undefined Optimizer")        
     end
    
end
