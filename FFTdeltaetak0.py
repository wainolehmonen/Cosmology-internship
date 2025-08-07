# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 13:41:53 2025

@author: Wäinö Lehmonen
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import random

# TODO: make all the codes similar i.e. update changes to all codes

# ---SETTINGS---

# constants (m/H <= 3/2 for real nu, for plotting pick m/H = small like 0.01)
m = 0.01  # mass of field
H = 1  # Hubble rate
d = 4  # dimension

k = 2  # norm of k bar in [0, inf)
eta = -1  # (etabar) mean of conformal times 1/2(eta' + eta) in (-inf, 0)

# number of data points for plotting and FFT
N = 1000

L = 15  # limits for k0etabar

# --------------

nu = np.sqrt(((d - 1)/2)**2 - (m/H)**2)

print('Scale factor a =', -1/(eta*H))

# integration variable u = Delta_eta/(2eta_bar) runs over (-1, 1)
epsilon = 1e-10  # small threshold in order to avoid limit points (nan)
urange = np.linspace(-1 + epsilon, 1 - epsilon, N)

etak0range = np.linspace(-k*eta - L, -k*eta + L, N)


def integrand_for_Delta(eta_k0, eta_bar, u):
    """
    This integrated over u in (-1, 1) should be Delta^<_k (times some constant)

    Parameters
    ----------
    k_0 : array-like
        frequency
    eta_bar : array-like
        mean of conformal times 1/2(eta' + eta)
    u : array-like
        Integration variable u = (Deltaeta)/etabar;
        Deltaeta = eta' - eta: conformal time difference

    Returns
    -------
    array-like
        Integrand for given values

    """
    hankel_a = sc.special.hankel1(nu, k*eta_bar*(u - 1))
    hankel_b = sc.special.hankel2(nu, -k*eta_bar*(u + 1))
    c = (1 - u**2)**(3/2)*np.exp(-1j*2*u*eta_k0)
    return np.pi/2*H**2*eta_bar**4*c*hankel_a*hankel_b


def Delta_benchmark_integral(eta_k_0, eta_bar):
    """
    Uses scipy quad to evaluate the integral for Delta at given points

    Parameters
    ----------
    k_0 : float
        frequency
    eta_bar : array-like
        mean of conformal times

    Returns
    -------
    array-like
        Delta at given points

    """
    I_Re = lambda u: integrand_for_Delta(eta_k_0, eta_bar, u).real
    I_Im = lambda u: integrand_for_Delta(eta_k_0, eta_bar, u).imag
    return sc.integrate.quad(I_Re, -1, 1)[0] \
        - 1j*sc.integrate.quad(I_Im, -1, 1)[0]


integral_etak = Delta_benchmark_integral(eta*k, eta)

# input matrix for FFT
x1, y1 = np.meshgrid(urange, etak0range)
# FFT matrix
FFTgrid = np.exp(-1j*x1*2*y1)*(urange[1]-urange[0])


# FFT using the grid defined above
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


# Plotting the FFT
Delta = FFT_for_G(eta, -urange)


plt.figure(figsize=(11, 6))

plt.axvline(0, c='gray')  # remove when |eta| >> 1 (out of plot range)
plt.axhline(0, c='gray')

plt.plot(etak0range, Delta, 'r', label=r'$\Delta^<_\bar k(k_0\bar{\eta})$ FFT')

plt.title(r'$\Delta^<_\bar k(k_0, \bar \eta)$ as a function of '
          r'$k_0|\bar \eta|$ when $\bar \eta={a},\ k={b}$'.format(a=eta, b=k),
          fontsize=14)
plt.xlabel(r'$k_0|\bar\eta|\in[k|\bar\eta|-L,k|\bar\eta|+L],\ L={a}$'
           .format(a=L), fontsize=13)

plt.axvline(-eta*k, c='b', linestyle='--')
plt.plot(-eta*k, integral_etak, 'bD',
         label=r'$\Delta^<_\bar{k}(k\bar{\eta})$')

# benchmarking for FFT
for i in range(5):
    randetak0 = random.choice(etak0range)
    plt.plot(randetak0, Delta_benchmark_integral(-randetak0, eta), 'yo')

plt.legend(loc='upper left', fontsize=14, shadow=True)
plt.grid()
plt.show()
