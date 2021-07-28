#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rand_direction(*shape):
    #Note: last dimension is dimensionality of the random unit vector.
    vecs = np.random.normal(0,1,size=shape)
    units = np.linalg.norm(vecs,axis=-1,keepdims=True)
    return vecs/units

class MultiscaleNDFunc():
    def __init__(self,min_freq,max_freq,n_frequencies,power,input_dim):
        self.n = n_frequencies
        self.phases = 2*np.pi *np.random.rand(self.n)
        self.freqs = (max_freq-min_freq)*np.random.rand(self.n)+min_freq
        self.freqs.sort()
        self.amplitudes = self.freqs**(-power) * np.random.normal(0,1,size=n_frequencies)
        self.directions = rand_direction(n_frequencies,input_dim)
    
    def __call__(self,x):
        args = np.dot(x,self.directions.T)
        raw = self.amplitudes*np.exp(2j*np.pi*self.freqs*args+1j*self.phases)
        real = np.real(raw)
        return real.sum()


if __name__ == '__main__':

    scale_max = 20
    n_funcs=100
    power=1

    n_examples = 5
    fig,axarr = plt.subplots(n_examples,1,figsize=(6,10))
    for ax in axarr:
        m = MultiscaleNDFunc(1,scale_max,n_funcs,power,1)
        x = np.linspace(0,1,10000)
        ev = np.asarray([m([xi]) for xi in x])
        plt.sca(ax)
        plt.plot(x,ev)
    plt.show()


    n_examples = 3
    L=300
    scale_max = 20
    n_funcs = 100
    power=1

    X,Y=np.meshgrid(np.linspace(0,1,L),np.linspace(0,1,L))
    inputs = np.stack([X,Y],axis=2)
    print(inputs.shape)
    for _ in range(n_examples):
        m = MultiscaleNDFunc(1,scale_max,n_funcs,power,2)
        ev = np.vectorize(m,signature="(m)->()")(inputs)
        flatx = X.reshape(L**2)
        flaty = Y.reshape(L**2)
        flatout = ev.reshape(L**2)
        plt.subplots()
        plt.scatter(flatx,flaty,c=flatout)
        plt.colorbar()
        fig,ax =plt.subplots(subplot_kw={"projection": "3d"})
        rs = lambda x: x.reshape(-1,L,L,)
        cax=ax.plot_surface(X,Y,ev,cmap=matplotlib.cm.coolwarm)
        ax.set_zticks(np.linspace(ev.min(),ev.max(),20))
        plt.colorbar(cax)
        plt.show()

