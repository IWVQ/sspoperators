# -*- coding: utf-8 -*-
"""
verificación numérica para el operador de prueba de existencia
"""

import math
import torch
from torch.utils.data import WeightedRandomSampler
import triton
import triton.language as tl
from math import *

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def kernel_Ts_operator(d_ptr, dx_ptr, N, S, BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(axis=0)
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N

    d = tl.load(d_ptr + offsets, mask=mask).cast(tl.int32)
    d_s = tl.load(d_ptr + (offsets - (S >> 1)), mask=mask).cast(tl.int32)
    
    d = tl.where(((offsets + 1) % S) == 0, d | d_s, d*0)

    tl.store(dx_ptr + offsets, d, mask=mask)

def run_circuit(w, n):
    N = 1 << n
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )

    zeros = torch.zeros(N, device=DEVICE, dtype=torch.float)
    ones = torch.full((N,), 1, device=DEVICE, dtype=torch.float)

    xState = torch.complex(zeros, zeros)
    dState = torch.zeros_like(zeros)

    # initialization
    xState = xState + 1/sqrt(N)
    # torch.cuda.synchronize()


    # simulation of Grover iterator for amplitude amplification
    Psi0 = xState # fully balanced
    for j in range(floor(pi/(4*asin(sqrt(1/N))))):
        # sign changing for N-1, (I - 2P)
        xState[N-1] = -xState[N-1]
        # Psi0 reflector operator simulation
        #Psi0_dot_xState = torch.mathmul(Psi0.unsqueeze(0), xState.unsqueeze(0).t()).t().squeeze()
        Psi0_dot_xState = torch.dot(Psi0, xState).item()
        xState = 2*Psi0_dot_xState*Psi0 - xState


    # ORy operator, amplitude amplifying for N-1
    #Ry = torch.tensor([[sqrt(2+sqrt(2))/2, -sqrt(2-sqrt(2))/2],
    #                   [sqrt(2-sqrt(2))/2,  sqrt(2+sqrt(2))/2]], 
    #                   device=DEVICE, dtype=torch.float)
    #Ry = torch.complex(Ry, torch.tensor([[0,0],[0,0]], device=DEVICE, dtype=torch.float))
    # Warning: Kronecker product eats a lot of memory :(
    #ORy = torch.tensor([[1]], device=DEVICE, dtype=torch.float) # init
    #for i in range(n):
    #    ORy = torch.kron(Ry, ORy)
    #xState = torch.matmul(ORy, xState.unsqueeze(0).t()).t().squeeze()
    # torch.cuda.synchronize()

    # ----

    # U operator
    iM = torch.arange(0,n,1, device=DEVICE, dtype=torch.int).expand(N,n).t()
    kM = torch.arange(0,N,1, device=DEVICE, dtype=torch.int).expand(n,N)
    wk = torch.matmul(w.unsqueeze(0).float(), ((kM >> iM) & 1).float()).squeeze(0)
    dState = torch.where(wk == 0, ones, zeros)
    # torch.cuda.synchronize()

    # T operator
    dNext = torch.zeros_like(dState)
    for s in range(1,n):
        S = 1 << s
        kernel_Ts_operator[grid](dState, dNext, N, S, BLOCK_SIZE=32)
        torch.cuda.synchronize()
        dState, dNext = dNext, dState

    # calculate weights
    weights = torch.real(xState*xState.conj())
    # torch.cuda.synchronize()

    # measure
    r = list(WeightedRandomSampler(weights=weights, num_samples=1, 
                                   replacement=True))[0]

    # display information
    d = int(dState[r].item())
    print(f'x_r |r,d_r〉 = {xState[r]}|{r},{d}〉; prob={weights[r]}')
    t = N-1
    delta = int(dState[t].item())
    print(f'x_(N-1) |N-1,δ〉 = {xState[t]}|{t},{delta}〉; prob={weights[t]}')
    print(f'system collapsed to: {bin(d)[2:] + bin(r)[2:]}, d={d}, r={r}')
    print(f'with probability: {weights[r]}')
    
    # output
    return d

def ssp_has_solution(w:[int]) :
    W = torch.tensor(w, device=DEVICE, dtype=torch.int)
    
    res = run_circuit(W, len(w))
    
    if res == 1:
        print("resultado: tiene al menos una solución")
    else: 
        print("resultado: no tiene solución")

if __name__ == '__main__':
    ssp_has_solution([1,-1,1,4,5,-6,2,8,0,-12,
                      1,-1,1,4,6,4,9,12,-34,45,
                      -21,123,43,12])
