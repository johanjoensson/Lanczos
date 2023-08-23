using LinearAlgebra
using SparseArrays
using BenchmarkTools

@enum Reort begin
    FullReort
    PartialReort
end

struct LanczosIt{T <: Union{Vector, Matrix}}
    A::Union{Hermitian, Symmetric}
    v₀::T
end

struct LanczosVectorIt{T <: Union{Vector, Matrix}}
    A::Union{Hermitian, Symmetric}
    v₀::T
    αs
    βs
end

mutable struct LanczosItStatus{T <: Union{Vector, Matrix}}
    β  ::Union{Real, UpperTriangular}
    vₚ ::T
    vₙ ::T
    i  ::Integer
end

mutable struct LanczosVectorItStatus{T <: Union{Vector, Matrix}}
    vₚ ::T
    vₙ ::T
    i  ::Integer
end

function Base.iterate(L::LanczosVectorIt{ <: Vector})
    local A = L.A
    local v = L.v₀
    if size(L.αs, 1) == 0
        return nothing
    end
    local α = L.αs[1]
    local vₙ = similar(v)

    mul!(vₙ, A, v)
    axpy!(-1, v*α, vₙ)

    v, LanczosVectorItStatus(v, normalize(vₙ), 2)
end

function Base.iterate(L::LanczosVectorIt{ <: Vector}, status::LanczosVectorItStatus)
    local vₚ = status.vₚ
    local v = status.vₙ
    local i = status.i
    local A = L.A
    if i > size(L.αs, 1)
        return nothing
    end

    local β = L.βs[i - 1]
    local α = L.αs[i]
    
    # println("i = $i")
    # println("--->αᵢ = $α, vᵢ'Avᵢ = $(dot(v, A, v))")

    # vₙ = Av - vα - vₚβ'
    # β = |vₙ|
    # vₙ = vₙ/|vₙ|
    local vₙ = similar(v)
    mul!(vₙ, A, v)
    axpy!(-1, v*α, vₙ)
    axpy!(-1, vₚ*β, vₙ)

    v, LanczosVectorItStatus(v, normalize(vₙ), i + 1)
end

function Base.iterate(L::LanczosVectorIt{ <: Matrix})
    local A = L.A
    local v = L.v₀
    if size(L.αs, 3) == 0
        return nothing
    end
    local α = L.αs[:, :, 1]
    local vₙ = similar(v)

    # vₙ = Av - vα
    # vₙ, β = qr(vₙ)
    mul!(vₙ, A, v)
    mul!(vₙ, v, α, -1, 1)
    local F = qr(vₙ)

    v, LanczosVectorItStatus(v, Matrix(F.Q), 1+size(v, 2))
end

function Base.iterate(L::LanczosVectorIt{ <: Matrix}, status::LanczosVectorItStatus)
    local vₚ = status.vₚ
    local v = status.vₙ
    local i = status.i
    local A = L.A
    if i > size(L.αs, 3)*size(L.αs, 1)
        return nothing
    end

    local β = UpperTriangular(L.βs[:, :, i ÷ size(v, 2)])
    local α 
    if eltype(A) <: Complex
        α = Hermitian(L.αs[:, :, Int(ceil(i / size(v, 2)))])
    else
        α = Symmetric(L.αs[:, :, Int(ceil(i / size(v, 2)))])
    end
    local vₙ = similar(v)
    mul!(vₙ, A, v)
    # vₙ = Av - vα - vₚβ'
    # vₙ, β = qr(vₙ)
    mul!(vₙ, v, α, -1, 1)
    mul!(vₙ, vₚ, β', -1, 1)
    local F = qr(vₙ)

    v, LanczosVectorItStatus(v, Matrix(F.Q), i + size(v, 2))
end

function Base.iterate(L::LanczosIt{ <: Matrix})
    local A = L.A
    local v = L.v₀
    if size(A, 1) == 0
        return nothing
    end
    local α = Matrix{eltype(A)}(undef, size(v, 2), size(v, 2))
    local vₙ = similar(v)

    # α = v' A v
    mul!(vₙ, A, v)
    mul!(α, v', vₙ)

    # vₙ = Av - vα
    # vₙ, β = qr(vₙ)
    mul!(vₙ, v, α, -1, 1)
    local F = qr(vₙ)

    (α, UpperTriangular(F.R), v), LanczosItStatus(UpperTriangular(F.R), v, Matrix(F.Q), 1+size(v, 2))
end

function Base.iterate(L::LanczosIt{ <: Matrix}, status::LanczosItStatus)
    local β = status.β
    local vₚ = status.vₚ
    local v = status.vₙ
    local i = status.i
    local A = L.A
    if i > size(A, 1)
        return nothing
    end

    local α = Matrix{eltype(A)}(undef, size(v, 2), size(v, 2))
    local vₙ = similar(v)
    # α = v' A v
    mul!(vₙ, A, v)
    mul!(α, v', vₙ)
    # vₙ = Av - vα - vₚβ'
    mul!(vₙ, v, α, -1, 1)
    mul!(vₙ, vₚ, β', -1, 1)
    # vₙ, β = qr(vₙ)
    local F = qr(vₙ)

    (α, UpperTriangular(F.R), v), LanczosItStatus(UpperTriangular(F.R), v, Matrix(F.Q), i + size(v, 2))
end

function Base.iterate(L::LanczosIt{ <: Vector})
    local A = L.A
    local v = L.v₀
    if size(A, 1) == 0
        return nothing
    end
    local α::Real = dot(v,A,v)
    # vₙ = Av - vα
    local vₙ = similar(v)
    mul!(vₙ, A, v)
    axpy!(-1, v*α, vₙ)

    local β = norm(vₙ)
    (α, β, v), LanczosItStatus(β, v, normalize!(vₙ), 2)
end

function Base.iterate(L::LanczosIt{ <: Vector}, status::LanczosItStatus)
    local β = status.β
    local vₚ = status.vₚ
    local v = status.vₙ
    local i = status.i
    local A = L.A
    if i > size(A, 1)
        return nothing
    end

    local α::Real = dot(v, A, v)
    local vₙ = similar(v)
    # vₙ = Av - vα - vₚβ
    mul!(vₙ, A, v)
    axpy!(-1, v*α, vₙ)
    axpy!(-1, vₚ*β, vₙ)

    (α, norm(vₙ), v), LanczosItStatus(norm(vₙ), v, normalize!(vₙ), i + 1)
end

function modified_reort!(q::Vector, Q::Matrix)
    for j = 1:size(Q, 2)
        # for i = j+1:size(Q, 2)
        #     Q[:, i] .-= dot(Q[:, j], Q[:, i])*Q[:, j]
        # end
        q .-= dot(Q[:, j], q)*Q[:, j]
        # normalize!(Q[:, i])
    end
end

function reort!(q::Union{Vector, Matrix}, Q::Matrix)
    q .-= Q * Q' * q
end

function is_reduced_rank(β)
    any(i -> abs(i) < eps(real(typeof(i))), diag(β))
end

function restore_full_rank!(q, β)
        mask = abs.(diag(β)) .< eps(real(typeof(eltype(A))))
        while sum(mask) > 0
            q[:, mask] = rand(eltype(A), size(q, 1), size(q, sum(mask)))
            for Qᵢ in LanczosVectorIt(A, v0, αs, βs)
                reort!(q, Qᵢ)
            end
            F = qr(q)
            β .= F.R
            mask = abs.(diag(β)) .< eps(real(typeof(eltype(A))))
        end
        q .= Matrix(F.Q)
        q, β
end

function restore_full_rank!(q, β, Q)
        mask = abs.(diag(β)) .< eps(real(typeof(eltype(A))))
        while sum(mask) > 0
            q[:, mask] = rand(eltype(A), size(q, 1), size(q, sum(mask)))
            reort!(q, Q)
            F = qr(q)
            β .= F.R
            mask = abs.(diag(β)) .< eps(real(typeof(eltype(A))))
        end
        q .= Matrix(F.Q)
        q, β
end

function lanczos(A::Union{Hermitian, Symmetric}, v0::Matrix, k = 6, tol = eps(Float64); return_eigenvectors = true)
    N = size(A, 1)
    n = size(v0, 2)

    αs = Array{eltype(A)}(undef, n, n, 0)
    βs = Array{eltype(A)}(undef, n, n, 0)

    for (α, β, _) in LanczosIt(A, v0)
        αs = cat(αs, reshape(α, n, n, 1), dims = 3)
        βs = cat(βs, reshape(β, n, n, 1), dims = 3)
        if converged(αs, βs, k, tol)
            break
        end
    end
    if !return_eigenvectors
        return αs, βs, nothing
    end

    Q = Array{eltype(A)}(undef, N, size(αs, 3)n)
    for (i, Qᵢ) in enumerate(LanczosVectorIt(A, v0, αs, βs))
        Q[:, (i-1)n + 1 : i*n] = Qᵢ
    end
    αs, βs, Q
end

function lanczos(A::Union{Hermitian, Symmetric}, v0::Matrix, reort::Reort, k = 6, tol = eps(Float64); return_eigenvectors = true)
    N = size(A, 1)
    n = size(v0, 2)

    Q = Array{eltype(A)}(undef, N, 0)
    αs = Array{eltype(A)}(undef, n, n, 0)
    βs = Array{eltype(A)}(undef, n, n, 0)

    it = LanczosIt(A, v0)
    next = iterate(it)
    while next != nothing
        (α, β, q), state = next
        state.vₙ = state.vₙ*state.β

        if reort == FullReort
            reort!(state.vₙ, Q)
            F = qr(state.vₙ)
            state.β = UpperTriangular(F.R)
            state.vₙ = Matrix(F.Q)
        end

        if is_reduced_rank(β)
            restore_full_rank(state.vₙ, state.β, Q)
        end

        αs = cat(αs, reshape(α, n, n, 1), dims = 3)
        βs = cat(βs, reshape(state.β, n, n, 1), dims = 3)
        Q = hcat(Q, q)
        if converged(αs, βs, k, tol)
            break
        end

        next = iterate(it, state)
    end
    αs, βs, return_eigenvectors ? Q : nothing
end

function lanczos(A::Union{Hermitian, Symmetric}, v0::Vector, reort::Reort, k = 6, tol = eps(Float64); return_eigenvectors = true)
    N = size(A, 1)

    Q = Array{eltype(A)}(undef, N, 0)
    αs = Array{Real}(undef, 0)
    βs = Array{Real}(undef, 0)

    it = LanczosIt(A, v0)
    next = iterate(it)
    while next != nothing
        (α, β, q), state = next

        if reort == FullReort
            state.vₙ *= state.β
            modified_reort!(state.vₙ, Q)
            state.β = norm(state.vₙ)
            normalize!(state.vₙ)
        end

        αs = push!(αs, α)
        βs = push!(βs, state.β)
        Q = hcat(Q, q)
        if converged(αs, βs, k, tol)
            break
        end

        next = iterate(it, state)
    end
    αs, βs, return_eigenvectors ? Q : nothing
end

function lanczos(A::Union{Hermitian, Symmetric}, v0::Vector, k = 6, tol = eps(Float64); return_eigenvectors = true)
    N = size(A, 1)

    Q = Array{eltype(A)}(undef, N, 0)
    αs = Array{Real}(undef, 0)
    βs = Array{Real}(undef, 0)

    for (α, β, _) in LanczosIt(A, v0)
        αs = push!(αs, α)
        βs = push!(βs, β)
        if converged(αs, βs, k, tol)
            break
        end
    end

    if !return_eigenvectors
        return αs, βs, nothing
    end

    Q = Matrix{eltype(A)}(undef, size(A, 1), size(αs, 1))
    for (i, Qᵢ) in enumerate(LanczosVectorIt(A, v0, αs, βs))
        Q[:, i] = Qᵢ
    end
    αs, βs, Q
end

function eigen(A::Union{Hermitian, Symmetric}; v0::Union{Vector, Matrix, Nothing}, k::Int = 6, tol::Real = 0, blocksize::Int = 1)
    if v0 isa Nothing
        v0 = rand(eltype(A), size(A, 1), blocksize)
    end
    if tol == 0
        tol = eps(real(eltype(A)))
    end
    if size(v0, 2) == 1
        v0 = reshape(v0, size(v0, 1))
    end

    αs, βs, Q = lanczos(A, v0, FullReort, k, tol)
    F = lanczos_eigen(αs, βs)
    Eigen(F.values[begin:begin+k-1], Q * F.vectors[:, begin:begin+k-1])
end

function eigvals(A::Union{Hermitian, Symmetric}; v0::Union{Vector, Matrix, Nothing}, k::Int = 6, tol::Real = 0, blocksize::Int = 1)
    if v0 isa Nothing
        v0 = rand(eltype(A), size(A, 1), blocksize)
    end
    if tol == 0
        tol = eps(real(eltype(A)))
    end
    if size(v0, 2) == 1
        v0 = reshape(v0, size(v0, 1))
    end

    αs, βs, _ = lanczos(A, v0, FullReort, k, tol, return_eigenvectors = false)
    lanczos_eigvals(αs, βs)[begin:begin+k-1]
end

function eigvecs(A::Union{Hermitian, Symmetric}; v0::Union{Vector, Matrix, Nothing}, k::Int = 6, tol::Real = 0, blocksize::Int = 1)
    if v0 isa Nothing
        v0 = rand(eltype(A), size(A, 1), blocksize)
    end
    if tol == 0
        tol = eps(real(eltype(A)))
    end
    if size(v0, 2) == 1
        v0 = reshape(v0, size(v0, 1))
    end

    αs, βs, _ = lanczos(A, v0, FullReort, k, tol, return_eigenvectors = false)
    lanczos_eigvecs(αs, βs)[begin:begin+k-1]
end

function length(L::LanczosIt)
    (size(L.A, 1) ÷ size(L.v₀, 2))
end

function length(L::LanczosVectorIt)
    (size(L.A, 1) ÷ size(L.v₀, 2))
end

function lanczos_vectors(A::Union{Hermitian, Symmetric}, v0::Vector, αs::Vector, βs::Vector)
    Q = Array{eltype(A)}(undef, N, size(αs, 1))
    for (i, q) in enumerate(LanczosVectorIt(A, v0, αs, βs))
        if i > size(Q, 2)
            break
        end
        Q[:, i] = q
    end
    Q
end

function lanczos_vectors(A::Union{Hermitian, Symmetric}, v0::Matrix, αs::Array{T, 3}, βs::Array{T, 3}) where T
    n = size(v0, 2)
    Q = Array{eltype(A)}(undef, N, size(αs, 3)*size(αs, 1))
    for (i, q) in enumerate(LanczosVectorIt(A, v0, αs, βs))
        if i*n > size(Q, 2)
            break
        end
        Q[:, (i-1)*n + 1 : i*n] = q
    end
    Q
end

function lanczos_banded(αs::Array{T, 3}, βs::Array{T, 3}) where T
    N = size(αs, 1) * size(αs, 3)
    Ap = zeros(T, (N, N))
    Ap[1:n, 1:n] .= αs[:, :, 1]
    for i=n+1:n:N-1
        Ap[i:i+n-1, i:i+n-1] .= αs[:, :, ceil(Int, i/n)]
        Ap[i:i+n-1, i-n:i-1] .= βs[:, :, ceil(Int, (i-n)/n)]
        Ap[i-n:i-1, i:i+n-1] .= βs[:, :, ceil(Int, (i-n)/n)]'
    end
    if T <: Complex
        Ap = Hermitian(Ap)
    else
        Ap = Symmetric(Ap)
    end
    Ap
end

function lanczos_banded(αs::Array{T, 1}, βs::Array{T, 1}) where T
    SymTridiagonal(αs, βs[begin:end - 1])
end

function lanczos_convergence(αs::Array{T, 1}, βs::Array{T, 1}, k::Integer) where T
    if size(αs, 1) < k
        return [1]
    end
    local s = lanczos_eigvecs(αs, βs)
    # print("Δ = $(abs.(βs[end]*s[end, begin:begin+k - 1]))\n")
    return abs.(βs[end]*s[end, begin:begin+k - 1])
end

function lanczos_convergence(αs::Array{T, 3}, βs::Array{T, 3}, k::Integer) where T
    if size(αs, 3) < k
        return [1]
    end
    local n = size(αs, 1)
    local s = lanczos_eigvecs(αs, βs)
    # print("r = $(βs[:, :, end]*s[end-n+1:end, begin:begin+k - 1])\n")
    # print("Δ = $(opnorm(βs[:, :, end]*s[end-n+1:end, begin:begin+k - 1], 2))\n")
    return opnorm(βs[:, :, end]*s[end-n+1:end, begin:begin+k - 1], 2)
end

function converged(αs, βs, k, δ)
    return all(lanczos_convergence(αs, βs, k) .< δ)
end

function lanczos_eigen(αs::Array{T, N}, βs::Array{T, N}) where T where N
    LinearAlgebra.eigen(lanczos_banded(αs, βs))
end

# function lanczos_eigen(αs::Array{T, 1}, βs::Array{T, 1}) where T
#     LinearAlgebra.eigen(lanczos_banded(αs, βs))
# end
# function lanczos_eigen(αs::Array{T, 3}, βs::Array{T, 3}) where T
#     LinearAlgebra.eigen(lanczos_banded(αs, βs))
# end

function lanczos_eigvals(αs::Array{T, N}, βs::Array{T, N}) where T where N
    LinearAlgebra.eigvals(lanczos_banded(αs, βs))
end

# function lanczos_eigvals(αs::Array{T, 1}, βs::Array{T, 1}) where T
#     LinearAlgebra.eigvals(lanczos_banded(αs, βs))
# end
# function lanczos_eigvals(αs::Array{T, 1}, βs::Array{T, 1}, irange::UnitRange) where T
#     LinearAlgebra.eigvals(SymTridiagonal(αs, βs[begin:end - 1]), irange)
# end

function lanczos_eigvecs(αs::Array{T, N}, βs::Array{T, N}) where T where N
    LinearAlgebra.eigvecs(lanczos_banded(αs, βs))
end

function lanczos_eigvecs(αs::Array{T, 1}, βs::Array{T, 1}) where T
    LinearAlgebra.eigvecs(SymTridiagonal(αs, βs[begin:end - 1]))
end

function lanczos_eigvecs(αs::Array{T, 1}, βs::Array{T, 1}, eigvals) where T
    LinearAlgebra.eigvecs(SymTridiagonal(αs, βs[begin:end - 1]), eigvals)
end


N = 500
n = 1
k = 10

exact = 1 ./ (1 .+ Array(1:N)) .+ 0.
# exact = Array(1:N) .+ 0.

A = spdiagm(0 => exact)
# A = spdiagm(0 => 1 ./ (1 .+ Array(1:N)))
A = Symmetric(A)
v0 = rand(Float64, N, n)
v0[:, 1] .= 1
v0 = Matrix(qr(v0).Q)
if size(v0, 2) == 1
    v0 =reshape(v0, size(v0, 1))
end

F = eigen(A, v0 = v0, k = k)
Fc = LinearAlgebra.eigen(Matrix(A))

print("λₑₓ :")
display(sort(exact)[1:k])
print("λ :")
display(F.values)

println("|Q' A Q - λI|ₘₐₓ = ", maximum(abs, F.vectors' * A * F.vectors - diagm(0 => sort(exact)[1:k])))
println("|λI - λₑₓ|ₘₐₓ = ", maximum(abs, sort(exact)[begin:begin+k-1] - F.values))
println("|Q'Q .- I|ₘₐₓ = $(maximum(abs, F.vectors'*F.vectors - I ))")

A = spdiagm(0 => exact .+ 0im)
# A = spdiagm(0 => 1 ./ (1 .+ Array(1:N)) .+ 0im)
A = Hermitian(A)
v0 = rand(ComplexF64, N, n)
v0[:, 1] .= 1
v0 = Matrix(qr(v0).Q)
if size(v0, 2) == 1
    v0 = reshape(v0, size(v0, 1))
end

F = eigen(A, v0 = v0, k = k)

print("λₑₓ :")
display(sort(exact)[begin:begin+k-1])
print("λ :")
display(F.values[begin:begin+k-1])

println("|Q' A Q - λI|ₘₐₓ = ", maximum(abs, F.vectors' * A * F.vectors - diagm(0 => sort(exact)[1:k])))
println("|λI - λₑₓ|ₘₐₓ = ", maximum(abs, sort(exact)[begin:begin+k-1] - F.values))
println("|Q'Q .- I|ₘₐₓ = $(maximum(abs, F.vectors'*F.vectors - I ))")

# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 300
# @benchmark _ = eigen(A, v0 = v0, k = k)

