__precompile__()
module InplaceRealFFTW

import Base: size, IndexStyle, getindex, setindex!, eltype, *, /, \, similar, copy, broadcast, real, complex

export AbstractPaddedArray, PaddedArray , plan_rfft!, rfft!, plan_irfft!, irfft!, rawreal

const Float3264 = Union{Float32,Float64}

abstract type AbstractPaddedArray{T,N,L} <: AbstractArray{Complex{T},N} end

struct PaddedArray{T<:Float3264,N,L} <: AbstractPaddedArray{T,N,L}
  c::Array{Complex{T},N} # Complex view of the array
  r::SubArray{T,N,Array{T,N},NTuple{N,UnitRange{Int64}},L} # Real view skipping padding
  rr::Array{T,N} # Raw real data, including padding

  function PaddedArray{T,N}(rr::Array{T,N},nx::Int) where {T<:Float3264,N}
    fsize = [size(rr)...]
    iseven(fsize[1]) || throw(ArgumentError("First dimension of allocated array must have even number of elements"))
    (nx == fsize[1]-2 || nx == fsize[1]-1) || throw(ArgumentError("Number of elements on the first dimension of array must be either 1 or 2 less than the number of elements on the first dimension of the allocated array"))
    fsize[1] = fsize[1]/2
    rsize = (fsize...)
    c = reinterpret(Complex{T}, rr, rsize)
    fsize[1] = nx
    r = view(rr,(1:l for l in fsize)...)
    return new{T,N,N==1?true:false}(c,r,rr)
  end # function

  function PaddedArray{T,N}(c::Array{Complex{T},N},nx::Int) where {T<:Float3264,N}
    fsize = [size(c)...]
    (iseven(nx) ? fsize[1]*2-2 == nx : fsize[1]*2-1 == nx) || throw(ArgumentError("The allocated array does not have the proper padding"))
    fsize[1] = fsize[1]*2
    rr = reinterpret(T, c, (fsize...))
    fsize[1] = nx
    r = view(rr,(1:l for l in fsize)...)
    return new{T,N,N==1?true:false}(c,r,rr)
  end # function

end # struct

PaddedArray(rr::Array{T,N},nx::Int) where {T,N} = PaddedArray{T,N}(rr,nx)
PaddedArray(c::Array{Complex{T},N},nx::Int) where {T,N} = PaddedArray{T,N}(c,nx)

@inline real(S::PaddedArray) = S.r
@inline complex(S::PaddedArray) = S.c
@inline rawreal(S::PaddedArray) = S.rr

size(S::AbstractPaddedArray) = size(complex(S))
IndexStyle(::Type{T}) where {T<:AbstractPaddedArray} = IndexLinear()
Base.@propagate_inbounds getindex(S::AbstractPaddedArray, i::Int) = getindex(complex(S),i)
Base.@propagate_inbounds getindex(S::AbstractPaddedArray{T,N,L}, I::Vararg{Int, N}) where {T,N,L} = getindex(complex(S),I...)
Base.@propagate_inbounds setindex!(S::AbstractPaddedArray,v,i::Int) =  setindex!(complex(S),v,i)
Base.@propagate_inbounds setindex!(S::AbstractPaddedArray{T,N,L},v,I::Vararg{Int,N}) where {T,N,L} =  setindex!(complex(S),v,I...)
eltype(S::AbstractPaddedArray{T,N,L}) where {T,N,L} = Complex{T} 
copy(S::AbstractPaddedArray) = PaddedArray(copy(complex(S)),size(real(S))[1])
similar(f::AbstractPaddedArray{T,N,L}) where {T,N,L} = PaddedArray{T,N}(similar(f.c),size(real(f))[1])
broadcast(op::Function, A::AbstractPaddedArray, other) = broadcast(op, complex(A), other)
broadcast(op::Function, other, A::AbstractPaddedArray) = broadcast(op, other, complex(A))
broadcast(op::Function, A::AbstractPaddedArray, other::AbstractPaddedArray) = broadcast(op, complex(other), complex(A))
broadcast(op::Function, A::AbstractPaddedArray) = broadcast(op, complex(A))

function PaddedArray(t::DataType,ndims::Vararg{Integer,N}) where N
  fsize = [ndims...]
  iseven(fsize[1]) ? fsize[1]+=2 : fsize[1]+=1
  a = Array{t,N}((fsize...))
  PaddedArray(a,ndims[1])
end

PaddedArray(t::DataType,ndims::NTuple{N,Integer}) where N = PaddedArray(t,ndims...)
PaddedArray(ndims::Vararg{Integer,N}) where N = PaddedArray(Float64,ndims)
PaddedArray(ndims::NTuple{N,Integer}) where N = PaddedArray(Float64,ndims...)

function PaddedArray(a::Array{T,N}) where {T<:Float3264,N}
  t = PaddedArray(T,size(a))
  t.r .= a 
  return t
end

###########################################################################################

function plan_rfft!(X::AbstractPaddedArray{T,N}, region;
                   flags::Integer=FFTW.ESTIMATE,
                   timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}

  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.ESTIMATE != 0
    p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(real(X), complex(X), region, flags, timelimit)
  else
    x = similar(X)
    p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(real(x), complex(x), region, flags, timelimit)
  end
  return p
end

plan_rfft!(f::AbstractPaddedArray;kws...) = plan_rfft!(f,1:ndims(f);kws...)

*(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = (A_mul_B!(complex(f),p,real(f));f)

rfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_rfft!(f,region) * f


##########################################################################################

function plan_brfft!(X::AbstractPaddedArray{T,N}, region;
                    flags::Integer=FFTW.PRESERVE_INPUT,
                    timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}
  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.PRESERVE_INPUT != 0
    a = similar(X)
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(complex(a), real(a), region, flags,timelimit)
  else
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(complex(X), real(X), region, flags,timelimit)
  end
end

function plan_irfft!(x::AbstractPaddedArray{T,N}, region; kws...) where {T,N}
  Base.DFT.ScaledPlan(plan_brfft!(x, region; kws...),Base.DFT.normalization(T, size(real(x)), region))
end

plan_irfft!(f::AbstractPaddedArray;kws...) = plan_irfft!(f,1:ndims(f);kws...)

*(p::Base.DFT.ScaledPlan,f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = begin
  A_mul_B!(real(f),p.p,complex(f))
  scale!(rawreal(f),p.scale)
  f
end

irfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_irfft!(f,region) * f

##########################################################################################

function /(f::AbstractPaddedArray{T,N},p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}) where {T<:Float3264,N}
  isdefined(p,:pinv) || (p.pinv = plan_irfft!(f,p.region))
  return p.pinv * f
end

function \(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N}
  isdefined(p,:pinv) || (p.pinv = plan_irfft!(f,p.region))
  return p.pinv * f
end

end # module
