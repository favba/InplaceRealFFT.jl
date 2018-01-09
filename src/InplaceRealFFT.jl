__precompile__()
module InplaceRealFFT

import Base: size, IndexStyle, getindex, setindex!, eltype, *,  \, similar, copy, real, read!

#For compatibility between Julia v0.6 and v0.7 - begin
if VERSION >= v"0.7-"
  import FFTW
  import AbstractFFTs
  using Base.@gc_preserve
else
  macro gc_preserve(s::Symbol,ex::Expr)
    return esc(ex)
  end
end
const M = VERSION >= v"0.7-" ? AbstractFFTs : Base.DFT
#For compatibility between Julia v0.6 and v0.7 - end

export AbstractPaddedArray, PaddedArray , plan_rfft!, rfft!, plan_irfft!, plan_brfft!, brfft!, irfft!

const Float3264 = Union{Float32,Float64}

abstract type AbstractPaddedArray{T,N} <: DenseArray{Complex{T},N} end

@eval struct PaddedArray{T<:Float3264,N,L} <: AbstractPaddedArray{T,N}
  r::SubArray{T,N,Array{T,N},NTuple{N,UnitRange{Int}},L} # Real view skipping padding
  ($(Symbol("#c")))::Array{Complex{T},N}

  function PaddedArray{T,N}(rr::Array{T,N},nx::Int) where {T<:Float3264,N}
    rrsize = size(rr)
    fsize = rrsize[1]
    iseven(fsize) || throw(ArgumentError("First dimension of allocated array must have even number of elements"))
    (nx == fsize-2 || nx == fsize-1) || throw(ArgumentError("Number of elements on the first dimension of array must be either 1 or 2 less than the number of elements on the first dimension of the allocated array"))
    fsize = fsize÷2
    csize = (fsize, rrsize[2:end]...)
    if VERSION >= v"0.7-" 
      @gc_preserve rr c = unsafe_wrap(Array{Complex{T},N},reinterpret(Ptr{Complex{T}},pointer(rr)),csize)
    else 
      c = reinterpret(Complex{T}, rr, csize)
    end
    rsize = (nx,rrsize[2:end]...)
    r = view(rr,(1:l for l in rsize)...)
    return  @gc_preserve rr new{T, N, N === 1 ? true : false}(r,c)
  end # function

end # struct

PaddedArray(rr::Array{T,N},nx::Int) where {T<:Float3264,N} = PaddedArray{T,N}(rr,nx)

@inline real(S::PaddedArray) = S.r
@inline unsafe_complex_view(S::PaddedArray) = getfield(S,Symbol("#c"))
copy(S::PaddedArray) = PaddedArray(copy(parent(real(S))),size(real(S))[1])
similar(f::PaddedArray,::Type{T},dims::Tuple{Vararg{Int,N}}) where {T, N} = PaddedArray{T}(dims) 
similar(f::PaddedArray{T,N,L},dims::NTuple{N2,Int}) where {T,N,L,N2} = PaddedArray{T}(dims) 
similar(f::PaddedArray,::Type{T}) where {T} = PaddedArray{T}(size(real(f))) 
similar(f::AbstractPaddedArray{T,N}) where {T,N} = PaddedArray{T,N}(similar(parent(real(f))),size(real(f))[1]) 

# AbstractPaddedArray interface
size(S::AbstractPaddedArray) = @gc_preserve S size(unsafe_complex_view(S))
IndexStyle(::Type{T}) where {T<:AbstractPaddedArray} = IndexLinear()
Base.@propagate_inbounds @inline getindex(S::AbstractPaddedArray, i::Int) = @gc_preserve S getindex(unsafe_complex_view(S),i)
Base.@propagate_inbounds @inline getindex(S::AbstractPaddedArray{T,N}, I::Vararg{Int, N}) where {T,N} = @gc_preserve S getindex(unsafe_complex_view(S),I...)
Base.@propagate_inbounds @inline setindex!(S::AbstractPaddedArray,v,i::Int) = @gc_preserve S setindex!(unsafe_complex_view(S),v,i)
Base.@propagate_inbounds @inline setindex!(S::AbstractPaddedArray{T,N},v,I::Vararg{Int,N}) where {T,N} = @gc_preserve S setindex!(unsafe_complex_view(S),v,I...)
# AbstractPaddedArray interface end

function PaddedArray{T}(ndims::Vararg{Integer,N}) where {T,N}
  fsize = (ndims[1]÷2 + 1)*2
  a = Array{T,N}((fsize,ndims[2:end]...))
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
    @gc_preserve X p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(real(X), unsafe_complex_view(X), region, flags, timelimit)
  else
    x = similar(X)
    @gc_preserve x p = FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(real(x), unsafe_complex_view(x), region, flags, timelimit)
  end
  return p
end

plan_rfft!(f::AbstractPaddedArray;kws...) = plan_rfft!(f,1:ndims(f);kws...)

*(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = (@gc_preserve f A_mul_B!(unsafe_complex_view(f),p,real(f));f)

rfft!(f::AbstractPaddedArray, region=1:ndims(f)) = plan_rfft!(f,region) * f


##########################################################################################

function plan_brfft!(X::AbstractPaddedArray{T,N}, region;
                    flags::Integer=FFTW.PRESERVE_INPUT,
                    timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:Float3264,N}
  (1 in region) || throw(ArgumentError("The first dimension must always be transformed"))
  if flags&FFTW.PRESERVE_INPUT != 0
    a = similar(X)
    return @gc_preserve a FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(unsafe_complex_view(a), real(a), region, flags,timelimit)
  else
    return @gc_preserve X FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(unsafe_complex_view(X), real(X), region, flags,timelimit)
  end
end

plan_brfft!(f::AbstractPaddedArray;kws...) = plan_brfft!(f,1:ndims(f);kws...)

*(p::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N},f::AbstractPaddedArray{T,N}) where {T<:Float3264,N} = (@gc_preserve f A_mul_B!(real(f),p,unsafe_complex_view(f)); real(f))

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
