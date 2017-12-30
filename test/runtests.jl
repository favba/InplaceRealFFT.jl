using InplaceRealFFTW
using Base.Test

VERSION >= v"0.7-" && using FFTW

a = rand(Float64,(8,4,4))
b = PaddedArray(a)
@test a == real(b)
@test rfft(a) ≈ rfft!(b) 
@test a ≈ irfft!(b)
@test rfft(a,1:2) ≈ rfft!(b,1:2) 
@test a ≈ irfft!(b,1:2)
@test rfft(a,(1,3)) ≈ rfft!(b,(1,3)) 
@test a ≈ irfft!(b,(1,3))

c = copy(b)
@test c == b
@test c.r == b.r
p = plan_rfft!(c)
@test p*c == rfft!(b)
@test p\c == irfft!(b)

@test typeof(similar(b)) === typeof(b)
@test size(similar(b,Float32)) === size(b)
@test size(similar(b,Float32).r) === size(b.r)
@test size(similar(b,(4,4,4)).r) === (4,4,4)
@test size(similar(b,Float32,(4,4,4)).r) === (4,4,4) 

a = rand(9,4,4)
b = PaddedArray(a)
@test a == real(b)
@test rfft(a) ≈ rfft!(b) 
@test a ≈ irfft!(b)
@test rfft(a,1:2) ≈ rfft!(b,1:2) 
@test a ≈ irfft!(b,1:2)
@test rfft(a,(1,3)) ≈ rfft!(b,(1,3)) 
@test a ≈ irfft!(b,(1,3))

for s in ((8,4,4),(9,4,4),(8,),(9,))
  a = rand(s)
  f = Base.Filesystem.tempname()
  write(f,a)
  @test a == real(PaddedArray(f,s,false))
  a = rand(Float32,s)
  write(f,a)
  @test a == real(PaddedArray{Float32}(f,s,false))
end

a = rand(4,4)
b = PaddedArray(a)
rfft!(b)
@test (brfft!(b) ./ 16) ≈ a