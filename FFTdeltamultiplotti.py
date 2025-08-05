# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:13:36 2025

@author: Wäinö Lehmonen
"""


import scipy as sc
import numpy as np
import matplotlib.pyplot as plt


# constants (m/H <= 3/2 for real nu; for plotting pick m/H = small like 0.01)
m = 0.01  # mass of field
H = 1  # Hubble rate
d = 4  # dimension

nu = np.sqrt(((d - 1)/2)**2 - (m/H)**2)

k = 1  # norm of k bar in [0, inf)

# number of data points for plotting and FFT
N = 1000
# integration variable u = Delta_eta/(2eta_bar) runs over (-1, 1)
epsilon = 1e-10  # small threshold in order to avoid limit points (nan)
urange = np.linspace(-1 + epsilon, 1 - epsilon, N)

# for variable k0
L = 10  # limits for k0etabar
R = 20
etak0range = np.linspace(-L, R, N)
# number of data points can be adjusted (then FFT matrix is no longer square)


def integrand_for_Delta(u, k0_eta_bar, eta_bar):
    """
    This integrated over u in (-1, 1) should be Delta^<_k (times some constant)

    Parameters
    ----------
    u : array-like
        Integration variable u = (Deltaeta)/etabar;
        Deltaeta = eta' - eta: conformal time difference
    k_0_eta_bar : array-like
        frequency times mean conformal time
    eta_bar : array-like
        mean of conformal times 1/2(eta' + eta)

    Returns
    -------
    array-like
        Integrand for given values

    """
    hankel_a = sc.special.hankel1(nu, k*eta_bar*(u - 1))
    hankel_b = sc.special.hankel2(nu, -k*eta_bar*(u + 1))
    c = (1 - u**2)**(3/2)*np.exp(-1j*2*u*k0_eta_bar)
    return np.pi/2*H**2*eta_bar**4*c*hankel_a*hankel_b


# Integral using scipy quad (real part (imaginary part is zero))
# for variable etak0
def Delta_benchmark_integral(eta_k_0, eta_bar):
    I_Re = sc.integrate.quad(integrand_for_Delta, -1, 1,
                             args=(eta_k_0, eta_bar))
    return I_Re[0]


# matrix for FFT
x1, y1 = np.meshgrid(urange, -etak0range)
FFTgrid = np.exp(-1j*x1*2*y1)*(urange[1]-urange[0])


# FFT using the above defined grid
def FFT_for_G(eta_bar, u):
    """
    Fourier transform of G wrt. u

    Parameters
    ----------
    eta_bar : array-like
        mean of conformal times 1/2(eta' + eta)
    u : array-like
        Integration variable u = Deltaeta/etabar;
        Deltaeta = eta' - eta: conformal time difference

    Returns
    -------
    array-like
        Fourier-transform for given values

    """
    hankel_a = sc.special.hankel1(nu, k*eta_bar*(u - 1))
    hankel_b = sc.special.hankel2(nu, -k*eta_bar*(u + 1))
    c = (1 - u**2)**(3/2)
    A = hankel_a*hankel_b*c
    return np.pi/2*H**2*eta_bar**4*np.matmul(FFTgrid, A)


def Deltaplot(etas):
    """
    Plots Delta for given etabars

    Parameters
    ----------
    etas : tuple
        etabars, etabar = (eta' + eta)/2 mean of conformal times

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 6))
    plt.axvline(x=0, c='gray')
    plt.axhline(y=0, c='gray')
    for i in range(len(etas)):
        Delta = FFT_for_G(etas[i], urange)
        benchmark_integral = Delta_benchmark_integral(etas[i]*k, etas[i])
        plt.plot(etak0range, Delta,
                 label=r'$\bar \eta = {a}$'.format(a=etas[i]))
        plt.axvline(-etas[i]*k, c='b', linestyle='dashed')
        plt.plot(-etas[i]*k, benchmark_integral, 'bD')
    plt.title(r'$\Delta^<_\bar k(k_0, \bar \eta)$ as a function of '
              r'$k_0\bar \eta$ when $k={a}$'.format(a=k),
              fontsize=14)
    plt.xlabel(r'$k_0|\bar\eta|\in[-L,R],\ L={a},\ R={b}$'.format(a=L, b=R),
               fontsize=13)
    plt.legend(loc='upper left', fontsize=14, shadow=True)
    plt.grid()
    plt.show()


Deltaplot([-1, -3])  # input etabar values (close together for good plot)
