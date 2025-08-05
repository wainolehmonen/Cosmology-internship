# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 08:34:40 2025

@author: Wäinö Lehmonen
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import random

# ---SETTINGS---

# constants (m/H <= 3/2 for real nu; for plotting pick m/H = small like 0.01)
m = 0.01  # mass of field
H = 1  # Hubble rate (H and etabar have inverse units, a = -1/(etabar*H))
d = 4  # dimension

k = 2  # norm of k bar in [0, inf)
etabar = -0.3  # mean of conformal times 1/2(eta' + eta) in (-inf, 0)
# -k*etabar > 1 subhorizon
# -k*etabar < 1 superhorizon

# number of data points for plotting and FFT (FFT matrix is N by N)
N = 1000

L = 15  # limits for k0 (when etabar = -1)

# --------------


nu = np.sqrt(((d - 1)/2)**2 - (m/H)**2)

print('Scale factor a =', -1/(etabar*H))

# integration variable u = Delta_eta/(2eta_bar) runs over (-1, 1)
epsilon = 1e-10  # small threshold in order to avoid limit points (nan)
urange = np.linspace(-1 + epsilon, 1 - epsilon, N)

# for variable k0
k0range = np.linspace(L/etabar + k, -L/etabar + k, N)
# TODO: FFT input, k0range depend on etabar, remove constant etabar and fix


def integrand_for_Delta(k_0, eta_bar, u):
    """
    This integrated over u in (-1, 1) should be Delta^<_k

    Parameters
    ----------
    k_0 : array-like
        frequency
    eta_bar : array-like
        mean of conformal times 1/2(eta' + eta)
    u : array-like
        Integration variable u = Deltaeta/etabar;
        Deltaeta = eta' - eta: conformal time difference

    Returns
    -------
    array-like
        Integrand for given values

    """
    hankel_a = sc.special.hankel1(nu, k*eta_bar*(u - 1))
    hankel_b = sc.special.hankel2(nu, -k*eta_bar*(u + 1))
    c = (1 - u**2)**(3/2)*np.exp(-1j*k_0*2*u*eta_bar)
    return np.pi/2*H**2*eta_bar**4*c*hankel_a*hankel_b


def Delta_benchmark_integral(k_0, eta_bar):
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
    I_Re = lambda u: integrand_for_Delta(k_0, eta_bar, u).real
    I_Im = lambda u: integrand_for_Delta(k_0, eta_bar, u).imag
    return sc.integrate.quad(I_Re, -1, 1)[0] \
        - 1j*sc.integrate.quad(I_Im, -1, 1)[0]


def plot_integrand(k_0, eta_bar):
    """
    Plots the integrand for delta as function of integration variable u

    Parameters
    ----------
    k_0 : array-like
        frequency
    eta_bar : float
        mean of conformal times 1/2(eta' + eta)

    Returns
    -------
    None.

    """
    # Plotting the integrand for fixed value k0
    integrand_k0 = integrand_for_Delta(k_0, eta_bar, urange)
    plt.figure()
    plt.plot(urange, integrand_k0, 'r')  # plots the real part of I
    # plt.plot(urange, integrand_k0.imag, 'b')  # imaginary part of I
    plt.title('Integrand as a function of integration variable u')
    plt.xlabel(r'$u=\frac{\Delta \eta}{2\bar{\eta}}$')
    plt.ylabel(r'$I(u, \bar{\eta}, k_0)$')
    plt.show()


# input matrix for FFT
x1, y1 = np.meshgrid(urange, k0range)


def FFT_for_G(eta_bar, u, FFTmatrix):
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
    return np.pi/2*H**2*eta_bar**4*np.matmul(FFTmatrix, A)


def plot_Delta(etas):
    """
    Plots Delta for given etas, own figure for each eta

    Parameters
    ----------
    etas : tuple
        etabars for plotting

    Returns
    -------
    None.

    """
    for i in range(len(etas)):
        FFTgrid = np.exp(-1j*x1*2*y1*etas[i])*(urange[1]-urange[0])
        Delta = FFT_for_G(etas[i], urange, FFTgrid)

        plt.figure(figsize=(11, 6))

        plt.plot(k0range, Delta, 'r',
                 label=r'$\Delta^<_\bar{k}(k_0, \bar{\eta})$ FFT')

        plt.title(r'$\Delta^<_\bar k(k_0, \bar \eta)$ as a function of $k_0$'
                  r' when $\bar \eta={a},\ k={b}$'.format(a=etas[i], b=k))
        plt.xlabel(r'$k_0 \in \left[k + L/\bar\eta, k - L/\bar\eta \right],\ L={a}$'
                   .format(a=L))

        plt.axvline(0, c='gray')  # comment out when |etabar| is large (out of range)
        plt.axhline(0, c='gray')

        plt.axvline(k, c='b', linestyle='--')
        plt.plot(k, Delta_benchmark_integral(k, etas[i]), 'bD',
                 label=r'$\Delta^<_\bar{k}(k, \bar{\eta})$')

        # benchmarking for FFT
        for j in range(5):
            randk0 = random.choice(k0range)
            plt.plot(randk0, Delta_benchmark_integral(randk0, etas[i]), 'go')

        plt.legend(loc='upper left', fontsize=13, shadow=True)
        plt.grid()
        plt.show()


plot_Delta([-1, -0.3])  # input etabar values
