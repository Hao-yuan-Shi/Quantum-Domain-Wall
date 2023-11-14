using ITensors
using LinearAlgebra
using Parameters;
using SpecialFunctions

include("base_utils.jl")
include("dmrg_solver.jl")

#=
These functions can be used to observe the time evolution of Domain Wall Ising Model
=#
@with_kw mutable struct DW_System
    # System Information
    n::Int = 10
    J::Real = 1
    Delta::Real = 1
    h::Real = 0
    dmr::Union{Float64, Int} = 0

    # First point of the right part
    cut_point::Int = div( n, 2 ) + 1
    
    # Time related Information
    dt::Float64
    ttotal::Union{Float64, Int}
    time_step::Int = 1
    time_scale = 0:dt * time_step: ttotal
    time_evol = 0:dt:ttotal
    
    # Wavefunction
    s = siteinds("S=1/2", n)
    ini = MPS(s, [i <= n/2 ? "Up" : "Dn" for i in 1:n])
    psis = Vector{ITensors.MPS}([ini])

    # data
    data = zeros(Float64, length(0:dt:ttotal), n)
    
    # Processing Instructions
    contour_val = []
    # heatmap_flag = false

    # heatmap
    hm_plt = plot()
end


# Time Evolution Solver
function TE_DomainWall_GetGate(sys::DW_System)
    N = sys.n;
    J = sys.J;
    Delta = sys.Delta;
    h = sys.h;
    δ = sys.dmr
    s = sys.s
    dt = sys.dt
    
    # Get Gate
    gates = ITensor[]
    # Spin 2-sites coupling gates
    for j in 1:(N - 1)
        s1 = s[j]
        s2 = s[j + 1]
        hj =J * Delta * op("Sz", s1) * op("Sz", s2) + 
            J *(1+(-1)^j*δ) * 1 / 2 * op("S+", s1) * op("S-", s2) + 
            J *(1+(-1)^j*δ) * 1 / 2 * op("S-", s1) * op("S+", s2);
        Gj = exp(-im * dt / 2 * hj);
        push!(gates, Gj);
    end
    append!(gates, reverse(gates));
    
    # Magnetic Field Term
    for j = 1:N
        hj = - h * op("Sz", s[j])
        Gj = exp(-im * dt * hj)
        push!(gates, Gj)
    end
    return gates;
end

function TE_DomainWall_Evol!(sys::DW_System; kwargs...)
    dt = sys.dt;
    ttotal = sys.ttotal;
    psi = sys.ini
    # Cutoff during gate applying is initialized by input
# returned value is a vector of wave function
    Gate = TE_DomainWall_GetGate(sys);
# Iteration to generate Evolved Wavefunction
    println("Domain Wall Time Evolution Δ = $(sys.Delta)");
    println("-"^length(0.0:dt:ttotal));
    for i in 1:length(sys.time_evol)-1
        print("*");
        psi = apply(Gate, sys.psis[i]; kwargs...);
        normalize!(psi);
        push!(sys.psis, psi);
    end
    println("*");
end


# This is the fucntion that solves the whole the megnatization of all sites at any time.
function TE_DomainWall_SzSolver!(sys::DW_System; kwargs...)
    # data of the heat map, each row is, at some time point, expectation values of Sz.
    for i in 1:sys.time_step:length(sys.time_evol)
        sys.data[i, :] = expect(sys.psis[i], "Sz")
    end
end


function TE_DomainWall_Heatmap!(sys::DW_System)
    sys.hm_plt = heatmap(1:sys.n, sys.time_scale, abs.(sys.data[1:sys.time_step:end, :]), c=cgrad([:white, :black]), 
        xlabel="N-axis", ylabel="time-axis", title="HeatMap")
    length(sys.contour_val) > 0 && contour!(sys.hm_plt, 1:sys.n, sys.time_scale, abs.(sys.data[1:sys.time_step:end, :]), 
        levels=sys.contour_val, linecolor=:black, linewidth=2)
    display(plot!(sys.hm_plt))
end

# plot the magnetization of all sites at time t
function TE_DomainWall_Mag_Obs(sys::DW_System, t::Union{Float64, Int}; kwargs...)
    if t > sys.ttotal
        println("Time exceeds scale")
        return
    end
    N = sys.n;
    t_ind = round(Int, (t) / (sys.time_step * sys.dt) + 1 ) #t_ind is the line which is cloest to the time input
    plot(1:sys.n, sys.data[(t_ind - 1) * sys.time_step + 1, :], xlabel="Site Index", ylabel="Magnetization", title="Magnetization at Time $t")
end

function TE_DomainWall_Mag_Flow(sys::DW_System; kwargs...)
    N = sys.n;
    cut_point = sys.cut_point;
    delta_mag = [sum(row[cut_point:end]) for row in eachrow(sys.data[1:sys.time_step:end, :])] .+ (N-cut_point+1)/2;
    plot!(sys.time_scale, delta_mag)
    xlabel!("time")
    ylabel!("ΔM")
    title!("Magnetization Flow")
    # ylims!(0,0.5)
end

# The Exact Solution of XX Model
function XX_Exact(N, t) # Only works for even N!!!!!!
    result = []
    for n = 1:div(N,2)
        push!(result, -0.5 * sum([besselj(i, t)^2 for i in 1-n:n-1]) )
    end
    return append!(-reverse(result),result)
end

function TE_DomainWall_MagDev(sys::DW_System)
    N = sys.n
    dt = sys.dt;
    ttotal = sys.ttotal;
    mag_dev = Vector{Float64}();
    for i = 1:sys.time_step:length(sys.time_evol)
        SzM_ED =  XX_Exact(N, (i-1)*dt);
        push!(mag_dev, maximum(  abs.(sys.data[i, :] - SzM_ED) ) );
    end
    plot!(sys.time_scale[2:end], mag_dev[2:end], yscale=:log10, label="dt = $(sys.dt)")

end