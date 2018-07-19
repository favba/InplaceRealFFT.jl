using InplaceRealFFT
using Base.Test

VERSION >= v"0.7-" && using FFTW

let a = rand(Float64,(8,4,4)), b = PaddedArray(a), c = copy(b)

@testset "PaddedArray creation" begin
  @test a == real(b)
  @test c == b
  @test c.r == b.r
  @test typeof(similar(b)) === typeof(b)
  @test size(similar(b,Float32)) === size(b)
  @test size(similar(b,Float32).r) === size(b.r)
  @test size(similar(b,(4,4,4)).r) === (4,4,4)
  @test size(similar(b,Float32,(4,4,4)).r) === (4,4,4) 
end

@testset "rfft! and irfft!" begin
  @test rfft(a) ≈ rfft!(b) 
  @test a ≈ irfft!(b)
  @test rfft(a,1:2) ≈ rfft!(b,1:2) 
  @test a ≈ irfft!(b,1:2)
  @test rfft(a,(1,3)) ≈ rfft!(b,(1,3)) 
  @test a ≈ irfft!(b,(1,3))
  
  p = plan_rfft!(c)
  @test p*c ≈ rfft!(b)
  @test p\c ≈ irfft!(b)

  a = rand(Float64,(9,4,4))
  b = PaddedArray(a)
  @test a == real(b)
  @test rfft(a) ≈ rfft!(b) 
  @test a ≈ irfft!(b)
  @test rfft(a,1:2) ≈ rfft!(b,1:2) 
  @test a ≈ irfft!(b,1:2)
  @test rfft(a,(1,3)) ≈ rfft!(b,(1,3)) 
  @test a ≈ irfft!(b,(1,3))
end

@testset "Read binary file to PaddedArray" begin
  for s in ((8,4,4),(9,4,4),(8,),(9,))
    aa = rand(Float64,s)
    f = Base.Filesystem.tempname()
    write(f,aa)
    @test aa == real(PaddedArray(f,s,false))
    aa = rand(Float32,s)
    write(f,aa)
    @test aa == real(PaddedArray{Float32}(f,s,false))
  end
end

@testset "brfft!" begin
  a = rand(Float64,(4,4))
  b = PaddedArray(a)
  rfft!(b)
  @test (brfft!(b) ./ 16) ≈ a
end

@testset "FFTW MEASURE flag" begin
  c = similar(b)
  p = plan_rfft!(c,flags=FFTW.MEASURE)
  p.pinv = plan_irfft!(c,flags=FFTW.MEASURE)
  c .= b 
  @test c == b
  @test p*c ≈ rfft!(b)
  @test p\c ≈ irfft!(b)
end

end

struct WrappedArray{T,N,L} <: AbstractPaddedArray{T,N}
  data::PaddedArray{T,N,L}
  str::String
  int::Int
end

@inline Base.real(a::WrappedArray) = real(a.data)
@inline InplaceRealFFT.complex_view(a::WrappedArray) = InplaceRealFFT.complex_view(a.data)

@testset "AbstractPaddedArray Interface" begin


  a = PaddedArray(rand(Float64,(8,8,8)))
  b = copy(a)

  field = WrappedArray(b,"test",10)

  @test rfft!(field) ≈ rfft!(a)
  @test irfft!(field) ≈ irfft!(a)

end
