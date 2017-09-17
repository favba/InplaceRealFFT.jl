using InplaceRealFFTW
using Base.Test

# write your own tests here
a = rand(8,4,4)
b = PaddedArray(a)
@test a == real(b)
@test rfft(a) ≈ rfft!(b) 
@test a ≈ irfft!(b)
@test rfft(a,1:2) ≈ rfft!(b,1:2) 
@test a ≈ irfft!(b,1:2)
@test rfft(a,(1,3)) ≈ rfft!(b,(1,3)) 
@test a ≈ irfft!(b,(1,3))

a = rand(9,4,4)
b = PaddedArray(a)
@test a == real(b)
@test rfft(a) ≈ rfft!(b) 
@test a ≈ irfft!(b)
@test rfft(a,1:2) ≈ rfft!(b,1:2) 
@test a ≈ irfft!(b,1:2)
@test rfft(a,(1,3)) ≈ rfft!(b,(1,3)) 
@test a ≈ irfft!(b,(1,3))