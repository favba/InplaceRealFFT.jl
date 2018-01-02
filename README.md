# InplaceRealFFT.jl

[![Build Status](https://travis-ci.org/favba/InplaceRealFFT.jl.svg?branch=master)](https://travis-ci.org/favba/InplaceRealFFT.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/b35ephpx1s1uj97m?svg=true)](https://ci.appveyor.com/project/favba/inplacerealfftw-jl)
[![Coverage Status](https://coveralls.io/repos/github/favba/InplaceRealFFT.jl/badge.svg?branch=master)](https://coveralls.io/github/favba/InplaceRealFFT.jl?branch=master)

This package provides in-place real-to-complex (`rfft!`) and complex-to-real (`irfft! or brfft!`) Fast Fourier transforms through the `PaddedArray` type and the FFTW library.
Those transformations require [padding](http://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format) of the real data, which is done automatically by the `PaddedArray` type.

## Quick start

```julia
using InplaceRealFFT

a = rand(8,8);

b = PaddedArray(a); #copy the contents of `a` and returns a PaddedArray.
 
b.r # returns the real view of the data. `real(b)` provides the same behaviour.
#8×8 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}:
# 0.93659   0.87329     0.346501   0.801544   0.941651  0.0877721  0.0995318    0.669844
# 0.109875  0.666899    0.838005   0.968509   0.717388  0.531444   0.0668872    0.117582
# 0.989863  0.606462    0.89497    0.915566   0.200812  0.290512   0.611392     0.541901
# 0.577207  0.498214    0.729158   0.399541   0.607058  0.111457   0.501753     0.714163
# 0.156463  0.380791    0.0988714  0.0588034  0.899444  0.766816   0.000694876  0.410209
# 0.255844  0.00797572  0.865057   0.695091   0.730696  0.666373   0.852273     0.0511616
# 0.193414  0.248292    0.175299   0.372205   0.846093  0.0418562  0.110176     0.0440493
# 0.625984  0.167037    0.926273   0.691699   0.977561  0.31093    0.53347      0.533091

b.r == real(b) == a
#true

rfft!(b) == rfft(a) # rfft! performs an inplace real-to-complex transformation on a PaddedArray.
#true

b # The PaddedArray acts the same way as the complex view of the data.
#5×8 InplaceRealFFT.PaddedArray{Float64,2,false}:
#  31.6573+0.0im       -2.90925-3.83939im     2.11563+1.72884im    …   2.11563-1.72884im   -2.90925+3.83939im
#  2.35205-2.5001im    -0.58157-4.85974im     1.20488+1.97291im       0.939899-0.487852im   1.97138+0.0276521im
# 0.445956+0.763535im  -1.10923+3.15186im    0.659003+0.0507059im      3.24141+1.76044im   0.469835+1.66771im
#  1.61721+3.54008im     1.1342+0.36402im   -0.856937-0.249445im      0.615628-2.54437im    2.37486+0.751845im
# -2.43398+0.0im        2.30783+0.466147im    3.53816-0.692175im       3.53816+0.692175im   2.30783-0.466147im

b.c # returns the complex view of the array, same as complex(b).
#5×8 Array{Complex{Float64},2}:
#  31.6573+0.0im       -2.90925-3.83939im     2.11563+1.72884im    …   2.11563-1.72884im   -2.90925+3.83939im
#  2.35205-2.5001im    -0.58157-4.85974im     1.20488+1.97291im       0.939899-0.487852im   1.97138+0.0276521im
# 0.445956+0.763535im  -1.10923+3.15186im    0.659003+0.0507059im      3.24141+1.76044im   0.469835+1.66771im
#  1.61721+3.54008im     1.1342+0.36402im   -0.856937-0.249445im      0.615628-2.54437im    2.37486+0.751845im
# -2.43398+0.0im        2.30783+0.466147im    3.53816-0.692175im       3.53816+0.692175im   2.30783-0.466147im
 
irfft!(b) # inplace complex-to-real transform, the function returns the real view.
#8×8 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}:
# 0.93659   0.87329     0.346501   0.801544   0.941651  0.0877721  0.0995318    0.669844
# 0.109875  0.666899    0.838005   0.968509   0.717388  0.531444   0.0668872    0.117582
# 0.989863  0.606462    0.89497    0.915566   0.200812  0.290512   0.611392     0.541901
# 0.577207  0.498214    0.729158   0.399541   0.607058  0.111457   0.501753     0.714163
# 0.156463  0.380791    0.0988714  0.0588034  0.899444  0.766816   0.000694876  0.410209
# 0.255844  0.00797572  0.865057   0.695091   0.730696  0.666373   0.852273     0.0511616
# 0.193414  0.248292    0.175299   0.372205   0.846093  0.0418562  0.110176     0.0440493
# 0.625984  0.167037    0.926273   0.691699   0.977561  0.31093    0.53347      0.533091
 
 b # See, it changed!
#5×8 InplaceRealFFT.PaddedArray{Float64,2,false}:
#   0.93659+0.109875im    0.87329+0.666899im     0.346501+0.838005im  …    0.0995318+0.0668872im   0.669844+0.117582im
#  0.989863+0.577207im   0.606462+0.498214im      0.89497+0.729158im        0.611392+0.501753im    0.541901+0.714163im
#  0.156463+0.255844im   0.380791+0.00797572im  0.0988714+0.865057im     0.000694876+0.852273im    0.410209+0.0511616im
#  0.193414+0.625984im   0.248292+0.167037im     0.175299+0.926273im        0.110176+0.53347im    0.0440493+0.533091im
# 0.0884275+0.0im       0.0960886+0.0im         -0.230356+0.0im            -0.141574+0.0im        0.0312508+0.0im
 
 p = plan_rfft!(b) # plan_rfft! precomputes the plan and is probably what you want to work with.
#FFTW in-place real-to-complex plan for 8×8 (1, 10)-strided array of Float64
#(rdft2-rank>=2/1
#  (rdft2-r2hc-direct-8-x8 "r2cf_8")
#  (dft-direct-8-x5 "n1fv_8_avx"))

p*b
#5×8 InplaceRealFFT.PaddedArray{Float64,2,false}:
#  31.6573+0.0im       -2.90925-3.83939im     2.11563+1.72884im    …   2.11563-1.72884im   -2.90925+3.83939im
#  2.35205-2.5001im    -0.58157-4.85974im     1.20488+1.97291im       0.939899-0.487852im   1.97138+0.0276521im
# 0.445956+0.763535im  -1.10923+3.15186im    0.659003+0.0507059im      3.24141+1.76044im   0.469835+1.66771im
#  1.61721+3.54008im     1.1342+0.36402im   -0.856937-0.249445im      0.615628-2.54437im    2.37486+0.751845im
# -2.43398+0.0im        2.30783+0.466147im    3.53816-0.692175im       3.53816+0.692175im   2.30783-0.466147im

p\b # This is actually doing this: (p.pinv = plan_irfft!(b); p.pinv*b)
#8×8 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}:
# 0.93659   0.87329     0.346501   0.801544   0.941651  0.0877721  0.0995318    0.669844
# 0.109875  0.666899    0.838005   0.968509   0.717388  0.531444   0.0668872    0.117582
# 0.989863  0.606462    0.89497    0.915566   0.200812  0.290512   0.611392     0.541901
# 0.577207  0.498214    0.729158   0.399541   0.607058  0.111457   0.501753     0.714163
# 0.156463  0.380791    0.0988714  0.0588034  0.899444  0.766816   0.000694876  0.410209
# 0.255844  0.00797572  0.865057   0.695091   0.730696  0.666373   0.852273     0.0511616
# 0.193414  0.248292    0.175299   0.372205   0.846093  0.0418562  0.110176     0.0440493
# 0.625984  0.167037    0.926273   0.691699   0.977561  0.31093    0.53347      0.533091
 
 p.pinv
#0.015625 * FFTW in-place complex-to-real plan for 5×8 array of Complex{Float64}
#(rdft2-rank>=2/1
#  (rdft2-hc2r-direct-8-x8 "r2cb_8")
#  (dft-direct-8-x5 "n1bv_8_sse2"))

bp = plan_brfft!(b) # you can also use the unormalized back-transform
#FFTW in-place complex-to-real plan for 5×8 array of Complex{Float64}
#(rdft2-rank>=2/1
#  (rdft2-hc2r-direct-8-x8 "r2cb_8")
#  (dft-direct-8-x5 "n1bv_8_sse2"))

bp*(p*b)
#8×8 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}:
# 59.9418   55.8905    22.1761   51.2988   60.2657   5.61741   6.37004    42.87
#  7.03201  42.6815    53.6323   61.9846   45.9129  34.0124    4.28078     7.52525
# 63.3512   38.8136    57.2781   58.5962   12.852   18.5928   39.1291     34.6817
# 36.9413   31.8857    46.6661   25.5706   38.8517   7.13326  32.1122     45.7064
# 10.0137   24.3706     6.32777   3.76342  57.5644  49.0763    0.0444721  26.2534
# 16.374     0.510446  55.3637   44.4858   46.7645  42.6479   54.5454      3.27434
# 12.3785   15.8907    11.2192   23.8211   54.1499   2.6788    7.05126     2.81916
# 40.063    10.6904    59.2814   44.2687   62.5639  19.8995   34.1421     34.1178

b.r ./= 64
#8×8 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}:
# 0.93659   0.87329     0.346501   0.801544   0.941651  0.0877721  0.0995318    0.669844
# 0.109875  0.666899    0.838005   0.968509   0.717388  0.531444   0.0668872    0.117582
# 0.989863  0.606462    0.89497    0.915566   0.200812  0.290512   0.611392     0.541901
# 0.577207  0.498214    0.729158   0.399541   0.607058  0.111457   0.501753     0.714163
# 0.156463  0.380791    0.0988714  0.0588034  0.899444  0.766816   0.000694876  0.410209
# 0.255844  0.00797572  0.865057   0.695091   0.730696  0.666373   0.852273     0.0511616
# 0.193414  0.248292    0.175299   0.372205   0.846093  0.0418562  0.110176     0.0440493
# 0.625984  0.167037    0.926273   0.691699   0.977561  0.31093    0.53347      0.533091

```

## Using `rfft!` and `irfft!` with custom types: `AbstractPaddedArray` interface.

The inplace FFT is available to any subtype of the `AbstractPaddedArray` type. One just need to implement methods `Base.real` and `Base.complex` for the custom type and `rfft!` and `irfft!` should readily work:

```julia
using InplaceRealFFT

struct MyCustomArray{T} <: AbstractPaddedArray{T,3}
  data::PaddedArray{T,3,false} 
  str::String
  int::Int
end

@inline Base.real(a::MyCustomArray) = real(a.data)
@inline Base.complex(a::MyCustomArray) = complex(a.data)

a = MyCustomArray(PaddedArray(rand(8,8,8)),"My String",100)

rfft!(a)

irfft!(a)
```