__precompile__()
module InplaceRealFFTW

import Base: size, IndexStyle, getindex, setindex!, eltype, *, /, \, similar, copy, broadcast, real, complex

export PaddedArray , plan_rfft!, rfft!, plan_irfft!, irfft!

const Float3264 = Union{Float32,Float64}

struct PaddedArray{T<:Float3264,N} <: AbstractArray{Complex{T},N}
  c::Array{Complex{T},N} # Complex view of the array
  r::SubArray # Real view skipping padding
  rr::Array{T,N} # Raw real data, including padding

  function PaddedArray{T,N}(rr::Array{T,N},nx::Int) where {T<:Float3264,N}
    fsize = [size(rr)...]
    iseven(fsize[1])|| throw(ArgumentError("First dimension of allocated array must have even number of elements"))
    (nx == fsize[1]-2 || nx == fsize[1]-1) || throw(ArgumentError("Number of elements on the first dimension of array must be either 1 or 2 less than the number of elements on the first dimension of the allocated array"))
    fsize[1] = fsize[1]/2
    rsize = (fsize...)
    c = reinterpret(Complex{T}, rr, rsize)
    fsize[1] = nx
    r = view(rr,(1:l for l in fsize)...)
    return new{T,N}(c,r,rr)
  end # function

  function PaddedArray{T,N}(c::Array{Complex{T},N},nx::Int) where {T<:Float3264,N}
    fsize = [size(c)...]
    (iseven(nx) ? fsize[1]*2-2 == nx : fsize[1]*2-1 == nx) || throw(ArgumentError("The allocated array does not have the proper padding"))
    fsize[1] = fsize[1]*2
    rr = reinterpret(T, c, (fsize...))
    fsize[1] = nx
    r = view(rr,(1:l for l in fsize)...)
    return new{T,N}(c,r,rr)
  end # function

end # struct

PaddedArray(rr::Array{T,N},nx::Int) where {T,N} = PaddedArray{T,N}(rr,nx)
PaddedArray(c::Array{Complex{T},N},nx::Int) where {T,N} = PaddedArray{T,N}(c,nx)

size(S::PaddedArray) = size(S.c)
IndexStyle(::Type{T}) where {T<:PaddedArray} = IndexLinear()
Base.@propagate_inbounds getindex(S::PaddedArray,I...) = S.c[I...]
Base.@propagate_inbounds setindex!(S::PaddedArray,v,I) =  setindex!(S.c,v,I)
eltype(S::PaddedArray{T,N}) where {T,N} = Complex{T} 
copy(S::PaddedArray) = PaddedArray(copy(S.c),size(S.r)[1])
similar(f::PaddedArray) = PaddedArray(eltype(f.r),size(f.r))
real(S::PaddedArray) = S.r
complex(S::PaddedArray) = S.c
broadcast(op::Function, A::PaddedArray, other) = broadcast(op, A.c, other)
broadcast(op::Function, other, A::PaddedArray) = broadcast(op, other, A.c)
broadcast(op::Function, A::PaddedArray, other::PaddedArray) = broadcast(op, other.c, A.c)
broadcast(op::Function, A::PaddedArray) = broadcast(op, A.c)

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

function plan_rfft!(X::PaddedArray{T,N}, region;
                   flags::Integer=FFTW.ESTIMATE,
                   timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}

  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.ESTIMATE != 0
    p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(X.r, X.c, region, flags, timelimit)
  else
    x = similar(X)
    p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(x.r, x.c, region, flags, timelimit)
  end
  return p
end

plan_rfft!(f::PaddedArray;kws...) = plan_rfft!(f,1:ndims(f.r);kws...)

*(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::PaddedArray{T,N}) where {T<:Float3264,N} = (A_mul_B!(f.c,p,f.r);f)

rfft!(f::PaddedArray, region=1:ndims(f.r)) = plan_rfft!(f,region) * f


##########################################################################################

function plan_brfft!(X::PaddedArray{T,N}, region;
                    flags::Integer=FFTW.PRESERVE_INPUT,
                    timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}
  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.PRESERVE_INPUT != 0
    a = similar(X)
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(a.c, a.r, region, flags,timelimit)
  else
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(X.c, X.r, region, flags,timelimit)
  end
end

function plan_irfft!(x::PaddedArray{T,N}, region; kws...) where {T,N}
  Base.DFT.ScaledPlan(plan_brfft!(x, region; kws...),Base.DFT.normalization(T, size(x.r), region))
end

plan_irfft!(f::PaddedArray;kws...) = plan_irfft!(f,1:ndims(f.c);kws...)

*(p::Base.DFT.ScaledPlan,f::PaddedArray{T,N}) where {T<:Float3264,N} = begin
  A_mul_B!(f.r,p.p,f.c)
  scale!(f.rr,p.scale)
  f
end

irfft!(f::PaddedArray, region=1:ndims(f.c)) = plan_irfft!(f,region) * f

##########################################################################################

function /(f::PaddedArray{T,N},p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}) where {T<:Float3264,N}
  isdefined(p,:pinv) || (p.pinv = plan_irfft!(f,p.region))
  return p.pinv * f
end

function \(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::PaddedArray{T,N}) where {T<:Float3264,N}
  isdefined(p,:pinv) || (p.pinv = plan_irfft!(f,p.region))
  return p.pinv * f
end

end # module
