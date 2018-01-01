__precompile__()
module InplaceRealFFT

import Base: size, IndexStyle, getindex, setindex!, eltype, *,  \, similar, copy, real, complex, read!

if VERSION >= v"0.7-"
  import FFTW
  import AbstractFFTs
end

const M = VERSION >= v"0.7-" ? AbstractFFTs : Base.DFT

export AbstractPaddedArray, PaddedArray , plan_rfft!, rfft!, plan_irfft!, plan_brfft!, brfft!, irfft!

const Float3264 = Union{Float32,Float64}

abstract type AbstractPaddedArray{T,N} <: DenseArray{Complex{T},N} end

struct PaddedArray{T<:Float3264,N,L} <: AbstractPaddedArray{T,N}
  c::Array{Complex{T},N} # Complex view of the array
  r::SubArray{T,N,Array{T,N},NTuple{N,UnitRange{Int}},L} # Real view skipping padding

  function PaddedArray{T,N}(c::Array{Complex{T},N},nx::Int) where {T<:Float3264,N}
    csize = size(c)
    fsize = csize[1]
    (iseven(nx) ? fsize*2-2 == nx : fsize*2-1 == nx) || throw(ArgumentError("The allocated array does not have the proper padding"))
    fsize = fsize*2
    rrsize = (fsize,csize[2:end]...)
    #rr = reinterpret(T, c, (fsize...))
    rr = unsafe_wrap(Array{T,N},reinterpret(Ptr{T},pointer(c)),rrsize)
    rsize = (nx,rrsize[2:end]...)
    r = view(rr,(1:l for l in rsize)...)
    return new{T, N, N === 1 ? true : false}(c,r)
  end # function

end # struct

PaddedArray(c::Array{Complex{T},N},nx::Int) where {T<:Float3264,N} = PaddedArray{T,N}(c,nx)

@inline real(S::PaddedArray) = S.r
@inline complex(S::PaddedArray) = S.c
copy(S::PaddedArray) = PaddedArray(copy(complex(S)),size(real(S))[1])
similar(f::PaddedArray,::Type{T},dims::Tuple{Vararg{Int64,N}}) where {T, N} = PaddedArray{T}(dims) 
similar(f::PaddedArray{T,N,L},dims::Tuple) where {T,N,L} = PaddedArray{T}(dims) 
similar(f::PaddedArray,::Type{T}) where {T} = PaddedArray{T}(size(real(f))) 
similar(f::PaddedArray{T,N,L}) where {T,N,L} = PaddedArray{T,N}(similar(f.c),size(real(f))[1]) 

# AbstractPaddedArray interface
size(S::AbstractPaddedArray) = size(complex(S))
IndexStyle(::Type{T}) where {T<:AbstractPaddedArray} = IndexLinear()
Base.@propagate_inbounds @inline getindex(S::AbstractPaddedArray, i::Int) = getindex(complex(S),i)
Base.@propagate_inbounds @inline getindex(S::AbstractPaddedArray{T,N}, I::Vararg{Int, N}) where {T,N} = getindex(complex(S),I...)
Base.@propagate_inbounds @inline setindex!(S::AbstractPaddedArray,v,i::Int) =  setindex!(complex(S),v,i)
Base.@propagate_inbounds @inline setindex!(S::AbstractPaddedArray{T,N},v,I::Vararg{Int,N}) where {T,N} =  setindex!(complex(S),v,I...)
# AbstractPaddedArray interface end

function PaddedArray{T}(ndims::Vararg{Integer,N}) where {T,N}
  fsize = ndims[1]÷2 + 1
  a = Array{Complex{T},N}((fsize,ndims[2:end]...))
  PaddedArray{T,N}(a,ndims[1])
end
PaddedArray{T}(ndims::NTuple{N,Integer}) where {T,N} = PaddedArray{T}(ndims...)
PaddedArray(ndims::Vararg{Integer,N}) where N = PaddedArray{Float64}(ndims...)
PaddedArray(ndims::NTuple{N,Integer}) where N = PaddedArray{Float64}(ndims...)

function PaddedArray{T}(a::AbstractArray{<:Real,N}) where {T<:Float3264,N}
  t = PaddedArray{T}(size(a))
  @inbounds copy!(t.r, a) 
  return t
end
PaddedArray(a::AbstractArray{<:Real}) = PaddedArray{Float64}(a)

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

plan_brfft!(f::AbstractPaddedArray;kws...) = plan_brfft!(f,1:ndims(f);kws...)

*(p::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = (A_mul_B!(real(f),p,complex(f)); real(f))

brfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_brfft!(f,region) * f

function plan_irfft!(x::AbstractPaddedArray{T,N}, region; kws...) where {T,N}
  M.ScaledPlan(plan_brfft!(x, region; kws...),M.normalization(T, size(real(x)), region))
end

plan_irfft!(f::AbstractPaddedArray;kws...) = plan_irfft!(f,1:ndims(f);kws...)

*(p::M.ScaledPlan,f::AbstractPaddedArray) = begin
  p.p * f
  scale!(parent(real(f)),p.scale)
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
