__precompile__()
module InplaceRealFFT

import Base: size, length, start, next, done, indices, linearindices, IndexStyle, getindex, setindex!, eltype, *,  \, similar, copy, real, read!

#For compatibility between Julia v0.6 and v0.7 - begin
@static if VERSION >= v"0.7-"
  import FFTW
  import AbstractFFTs
  import LinearAlgebra: mul!
  using LinearAlgebra
else
  mul!(x,B,y) = A_mul_B!(x,B,y)
  copyto!(dts,src) = copy!(dts,src)
  rmul!(B,x) = scale!(B,x)
end
const M = @static VERSION >= v"0.7-" ? AbstractFFTs : Base.DFT
#For compatibility between Julia v0.6 and v0.7 - end

export AbstractPaddedArray, PaddedArray , plan_rfft!, rfft!, plan_irfft!, plan_brfft!, brfft!, irfft!, data

const Float3264 = Union{Float32,Float64}

abstract type AbstractPaddedArray{T,N} <: DenseArray{Complex{T},N} end

struct PaddedArray{T<:Float3264,N,L} <: AbstractPaddedArray{T,N}
  data::Array{T,N}
  r::SubArray{T,N,Array{T,N},NTuple{N,UnitRange{Int}},L} # Real view skipping padding
  c::(@static VERSION >= v"0.7-" ? Base.ReinterpretArray{Complex{T},N,T,Array{T,N}} : Array{Complex{T},N})

  function PaddedArray{T,N}(rr::Array{T,N},nx::Int) where {T<:Float3264,N}
    rrsize = size(rr)
    fsize = rrsize[1]
    iseven(fsize) || throw(ArgumentError("First dimension of allocated array must have even number of elements"))
    (nx == fsize-2 || nx == fsize-1) || throw(ArgumentError("Number of elements on the first dimension of array must be either 1 or 2 less than the number of elements on the first dimension of the allocated array"))
    fsize = fsize÷2
    csize = (fsize, rrsize[2:end]...)
    c = @static VERSION >= v"0.7-" ? reinterpret(Complex{T}, rr) : reinterpret(Complex{T}, rr, csize) 
    rsize = (nx,rrsize[2:end]...)
    r = view(rr,(1:l for l in rsize)...)
    return  new{T, N, N === 1 ? true : false}(rr,r,c)
  end # function

end # struct

PaddedArray(rr::Array{T,N},nx::Int) where {T<:Float3264,N} = PaddedArray{T,N}(rr,nx)

@inline real(S::PaddedArray) = S.r
@inline complex_view(S::PaddedArray) = S.c
@inline data(S::PaddedArray) = S.data
copy(S::PaddedArray) = PaddedArray(copy(parent(real(S))),size(real(S))[1])
similar(f::PaddedArray,::Type{T},dims::Tuple{Vararg{Int,N}}) where {T, N} = PaddedArray{T}(dims) 
similar(f::PaddedArray{T,N,L},dims::NTuple{N2,Int}) where {T,N,L,N2} = PaddedArray{T}(dims) 
similar(f::PaddedArray,::Type{T}) where {T} = PaddedArray{T}(size(real(f))) 
similar(f::AbstractPaddedArray{T,N}) where {T,N} = PaddedArray{T,N}(similar(parent(real(f))),size(real(f))[1]) 

# AbstractPaddedArray interface

@inline data(S::AbstractPaddedArray) = parent(real(S))

# iteration
@inline start(A::AbstractPaddedArray) = 1
@inline next(A::AbstractPaddedArray, i) = @inbounds begin return (A[i], i+1) end
@inline done(A::AbstractPaddedArray, i) = i > length(A) ? true : false

# size
@inline length(A::AbstractPaddedArray) = length(data(A))÷2

@inline @generated function size(A::AbstractPaddedArray{T,N}) where {T,N}
  r = Expr(:tuple)
  push!(r.args,:(Asize[1]÷2))
  for i=2:N
    push!(r.args,:(Asize[$i]))
  end
  quote
    Asize = size(data(A))
    return $r
  end
end

# indexing
@inline function getindex(A::AbstractPaddedArray{T,N}, i::Integer) where {T,N}
  d = data(A)
  @boundscheck checkbounds(d,2i)
  @inbounds begin 
    return Complex{T}(d[2i-1],d[2i])
  end
end

@inline @generated function getindex(A::AbstractPaddedArray{T,N}, I2::Vararg{Integer,N}) where {T,N}
  ip = :(2*I2[1])
  t = Expr(:tuple)
  for i=2:N
    push!(t.args,:(I2[$i]))
  end
  quote
    d = data(A)
    i = $ip
    I = $t
    @boundscheck checkbounds(d,i,I...)
    @inbounds begin 
      return Complex{T}(d[i-1,I...],d[i,I...])
    end
  end
end

@inline function setindex!(A::AbstractPaddedArray{T,N},x, i::Integer) where {T,N}
  d = data(A)
  @boundscheck checkbounds(d,2i)
  @inbounds begin 
    d[2i-1] = real(x)
    d[2i] = imag(x)
  end
  A
end

@inline @generated function setindex!(A::AbstractPaddedArray{T,N}, x, I2::Vararg{Integer,N}) where {T,N}
  ip = :(2*I2[1])
  t = Expr(:tuple)
  for i=2:N
    push!(t.args,:(I2[$i]))
  end
  quote
    d = data(A)
    i = $ip
    I = $t
    @boundscheck checkbounds(d,i,I...)
    @inbounds begin 
      d[i-1,I...] = real(x)
      d[i,I...] = imag(x)
    end
    A
  end
end

@inline @generated function indices(A::AbstractPaddedArray{T,N}) where {T,N}
  r = Expr(:tuple)
  push!(r.args,:(Base.OneTo(Asize[1]÷2)))
  for i=2:N
    push!(r.args,:(Base.OneTo(Asize[$i])))
  end
  quote
    Asize = size(data(A))
    return $r
  end
end

@inline linearindices(A::AbstractPaddedArray) = Base.OneTo(length(A))

IndexStyle(::Type{T}) where {T<:AbstractPaddedArray} = IndexLinear()

Base.unsafe_convert(::Type{Ptr{Complex{T}}},A::AbstractPaddedArray{T,N}) where {T,N} = convert(Ptr{Complex{T}},pointer(data(A)))

# AbstractPaddedArray interface end

function PaddedArray{T}(ndims::Vararg{Integer,N}) where {T,N}
  fsize = (ndims[1]÷2 + 1)*2
  a = zeros(T,(fsize,ndims[2:end]...))
  PaddedArray{T,N}(a,ndims[1])
end
PaddedArray{T}(ndims::NTuple{N,Integer}) where {T,N} = PaddedArray{T}(ndims...)
PaddedArray(ndims::Vararg{Integer,N}) where N = PaddedArray{Float64}(ndims...)
PaddedArray(ndims::NTuple{N,Integer}) where N = PaddedArray{Float64}(ndims...)

function PaddedArray{T}(a::AbstractArray{<:Real,N}) where {T<:Float3264,N}
  t = PaddedArray{T}(size(a))
  @inbounds copyto!(t.r, a) 
  return t
end
PaddedArray(a::AbstractArray{<:Real}) = PaddedArray{Float64}(a)

###########################################################################################

function plan_rfft!(X::AbstractPaddedArray{T,N}, region;
                   flags::Integer=FFTW.ESTIMATE,
                   timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}

  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.ESTIMATE != 0
    p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(real(X), complex_view(X), region, flags, timelimit)
  else
    x = similar(X)
    p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(real(x), complex_view(x), region, flags, timelimit)
  end
  return p
end

plan_rfft!(f::AbstractPaddedArray;kws...) = plan_rfft!(f,1:ndims(f);kws...)

*(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = (mul!(complex_view(f),p,real(f));f)

rfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_rfft!(f,region) * f


##########################################################################################

function plan_brfft!(X::AbstractPaddedArray{T,N}, region;
                    flags::Integer=FFTW.PRESERVE_INPUT,
                    timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}
  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.PRESERVE_INPUT != 0
    a = similar(X)
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(complex_view(a), real(a), region, flags,timelimit)
  else
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(complex_view(X), real(X), region, flags,timelimit)
  end
end

plan_brfft!(f::AbstractPaddedArray;kws...) = plan_brfft!(f,1:ndims(f);kws...)

*(p::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = (mul!(real(f),p,complex_view(f)); real(f))

brfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_brfft!(f,region) * f

function plan_irfft!(x::AbstractPaddedArray{T,N}, region; kws...) where {T,N}
  M.ScaledPlan(plan_brfft!(x, region; kws...),M.normalization(T, size(real(x)), region))
end

plan_irfft!(f::AbstractPaddedArray;kws...) = plan_irfft!(f,1:ndims(f);kws...)

*(p::M.ScaledPlan,f::AbstractPaddedArray) = begin
  p.p * f
  rmul!(parent(real(f)),p.scale)
  real(f)
end

irfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_irfft!(f,region) * f

##########################################################################################

function \(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N}
  isdefined(p,:pinv) || (p.pinv = plan_irfft!(f,p.region))
  return p.pinv * f
end

##########################################################################################

function read!(stream::IO, field::PaddedArray{T,N,L}, padded::Bool) where {T,N,L}
  rr = parent(field.r)
  if padded
    read!(stream,rr)
  else
    dims = size(real(field))
    nx = dims[1]
    nb = sizeof(T)*nx
    npencils = prod(dims)÷nx
    npad = iseven(nx) ? 2 : 1
    for i=0:(npencils-1)
      unsafe_read(stream,Ref(rr,Int((nx+npad)*i+1)),nb)
    end
  end
  return field
end

function read!(file::AbstractString, field::PaddedArray, padded::Bool)
  open(file) do io 
   return read!(io,field,padded) 
  end
end

function PaddedArray{T}(stream,dims, padded::Bool) where T
  field = PaddedArray{T}(dims)
  return read!(stream,field,padded)
end

function PaddedArray(stream,dims, padded::Bool)
  field = PaddedArray(dims)
  return read!(stream,field,padded)
end

end # module
