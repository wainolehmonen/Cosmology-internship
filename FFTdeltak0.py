# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 07:42:21 2025

@author: Wäinö Lehmonen
"""

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import random


# ---SETTINGS---

# constants (m/H <= 3/2 for real nu; for plotting pick m/H = small like 0.01)
m = 0.01  # mass of field
H = 1  # Hubble rate
d = 4  # dimension

k = 5  # norm of k bar in [0, inf)

# number of data points for plotting and FFT (FFT matrix is N by N)
N = 1000

L = 50  # limits for k0 (when eta = -1)

# --------------


nu = np.sqrt(((d - 1)/2)**2 - (m/H)**2)

# integration variable u = Delta_eta/(2eta_bar) runs over (-1, 1)
epsilon = 1e-10  # small threshold in order to avoid limit points (nan)
urange = np.linspace(-1 + epsilon, 1 - epsilon, N)


# for variable k0
def k0range(eta_bar):
    return np.linspace(-L/abs(eta_bar) + k, L/abs(eta_bar) + k, N)


def integrand_for_Delta(u, k_0, eta_bar):
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


# Integral using scipy quad (only real part (imaginary part is zero))
def Delta_benchmark_integral(k_0, eta_bar):
    """
    Uses scipy quad to evaluate the integral for Delta at given points

    Parameters
    ----------
    k_0 : float
        frequency
    eta_bar : float
        mean of conformal times

    Returns
    -------
    float
        Delta at given points

    """
    return sc.integrate.quad(integrand_for_Delta, -1, 1,
                             args=(k_0, eta_bar))[0]


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


# FFT matrix
def grid_for_FFT(eta_bar):
    # input matrix for FFT
    x1, y1 = np.meshgrid(urange, k0range(eta_bar))
    return np.exp(-1j*x1*2*y1*eta_bar)*(urange[1] - urange[0])


# FFT using the above defined grid
def FFT_for_G(eta_bar, u, gridmatrix):
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
    return np.pi/2*H**2*eta_bar**4*np.matmul(gridmatrix, A)


def deltaDelta(eta_bar, D, newk0range, newDelta, norm, omega_k):
    """
    Gives the quantity delta_Delta

    Parameters
    ----------
    eta_bar : float
        mean of conformal times (eta' + eta)/2
    D : float
        parameter Delta (not same Delta), gives boundaries for integral
    newk0range : array-like
        k0 values for integration
    newDelta : array-like
        Delta values, symmetric around max
    norm : float
        norm factor, integral of Delta over all k0
    omega_k : float
        normed integral of k0*Delta over all k0

    Returns
    -------
    array-like
        quantity delta_Delta as a function of Delta (D)

    """
    leftlimitindex = np.argmin(abs(newk0range - (omega_k - D)))
    rightlimitindex = np.argmin(abs(newk0range - (omega_k + D)))
    # k0range_cutright = newk0range[:rightlimitindex]
    # k0range_cut = k0range_cutright[leftlimitindex:]  # plottausta varten
    Delta_cutright = newDelta[:rightlimitindex]
    Delta_cut = Delta_cutright[leftlimitindex:]
    return sum(Delta_cut)*(newk0range[1] - newk0range[0])/norm
    # /norm for normed


def plot_Delta_and_deltaDelta(etas, plotting=True):

    omegas_k = np.empty(len(etas))

    for j in range(len(etas)):

        Delta = FFT_for_G(etas[j], urange, grid_for_FFT(etas[j]))

        # cut the range to be symmetric around max of Delta
        newk0range = k0range(etas[j])[0:2*np.argmax(Delta)]

        Drange = np.linspace(0, newk0range[-1], num=1000)
        newDelta = Delta[0:2*np.argmax(Delta)].real
        norm = sum(newDelta)*(newk0range[1] - newk0range[0])
        omega_k = sum(newk0range*newDelta)*(newk0range[1] - newk0range[0])/norm
        omegas_k[j] = omega_k
        deltajono = np.empty(len(Drange))
        for i in range(len(Drange)):
            deltajono[i] = deltaDelta(etas[j], Drange[i], newk0range, newDelta,
                                      norm, omega_k)

        if plotting:
            plt.figure(figsize=(11, 6))

            # print('Deltan maksimi k_0 =', k0range(etas[j])[np.argmax(Delta)])

            plt.plot(k0range(etas[j]), Delta, 'r',
                     label=r'$\Delta^<_\bar{k}(k_0, \bar{\eta})$ FFT')
            a = -1/(etas[j]*H)

            plt.title(r'$\Delta^<_\bar k(k_0, \bar \eta)$'
                      r' as a function of $k_0$'
                      r' when $\bar \eta={b},\ k={k},\ a={c},\ H={H}$'
                      .format(b=etas[j], k=k, c=a, H=H),
                      fontsize=14)
            plt.xlabel(r'$k_0 \in \left[k+L/\bar\eta,k-L/\bar\eta \right]$, '
                       '$L={L}$'.format(L=L), fontsize=13)

            # axes
            plt.axvline(0, c='gray')  # comment when |eta| >> 1 (out of range)
            plt.axhline(0, c='gray')

            plt.axvline(k, c='b', linestyle='--')
            plt.plot(k, Delta_benchmark_integral(k, etas[j]), 'bD',
                     label=r'$\Delta^<_\bar{k}(k_0=k, \bar{\eta})$')

            # benchmarking for FFT
            for i in range(5):
                randk0 = random.choice(k0range(etas[j]))
                plt.plot(randk0, Delta_benchmark_integral(randk0,
                                                          etas[j]), 'go')

            plt.legend(loc='upper left', fontsize=14, shadow=True)
            plt.grid()
            plt.show()

            plt.figure(figsize=(11, 6))

            plt.plot(Drange, deltajono, c='g')

            plt.ylabel(r'$\delta_\Delta$', fontsize=13)
            plt.xlabel(r'$\Delta$ in units of $k_0$', fontsize=13)
            plt.title(r'$\delta_\Delta$ when '
                      r'$\bar \eta={b},\ k={k},\ a={a},\ H={H}$'
                      .format(b=etas[j], k=k, a=a, H=H), fontsize=14)

            plt.grid()
            plt.show()

    return omegas_k


def plot_everything(etas, dotplot=True, plottingdeltas=True):

    omegas_k = plot_Delta_and_deltaDelta(etas, plottingdeltas)

    # dot-plot how omega_k changes as a function of eta
    plt.figure(figsize=(11, 6))
    plt.title(r'$\omega_k(\bar\eta)$', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    plt.ylabel(r'$\omega_k$', fontsize=13)
    if dotplot:
        plt.plot(etas, omegas_k, 'bx')
    else:
        plt.plot(etas, omegas_k, 'b')
    plt.grid()
    plt.show()


# input etabar values
etarange = [-0.05, -0.1, -0.2, -0.5, -1]
# etarange = np.linspace(-0.1, -10, 100)
# (etabar) mean of conformal times (eta' + eta)/2 in (-inf, 0)
# -k*eta > 1 subhorizon
# -k*eta < 1 superhorizon

# when number of etas is large, dotplot=False, plottingdeltas=False
plot_everything(etarange)

# when m/H -> 3/2 something weird close to eta = zero, probably grid related

# TODO: MUOKKAA KOODI LUETTAVAKSI, lisää dokumentaatiot joka funktioon
# TODO: selkeät ohjeet, että miten ajetaan
# TODO: LEGENDI jossa vakiot listattu, joka kuvaan!!!
