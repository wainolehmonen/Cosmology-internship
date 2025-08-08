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
eta = -0.1  # (etabar) mean of conformal times 1/2(eta' + eta) in (-inf, 0)
# -k*eta > 1 subhorizon
# -k*eta < 1 superhorizon

# number of data points for plotting and FFT (FFT matrix is N by N)
N = 4000

L = 100  # limits for k0 (when eta = -1)

# --------------


nu = np.sqrt(((d - 1)/2)**2 - (m/H)**2)

print('Scale factor a =', -1/(eta*H))

# integration variable u = Delta_eta/(2eta_bar) runs over (-1, 1)
epsilon = 1e-10  # small threshold in order to avoid limit points (nan)
urange = np.linspace(-1 + epsilon, 1 - epsilon, N)

# for variable k0
k0range = np.linspace(L/eta + k, -L/eta + k, N)


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


integral_k = Delta_benchmark_integral(k, eta)
# print(benchmark_integral)

integrand_k = integrand_for_Delta(k, eta, urange)  # integrand at points u
# # this benchmark (for FFT) uses same u grid as the FFT (compare with scipy)
# benchmark_integral = sum(integrand_k)*(urange[1]-urange[0])
# print(benchmark_integral)


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
# FFT matrix
FFTgrid = np.exp(-1j*x1*2*y1*eta)*(urange[1] - urange[0])


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


# Delta at points k0range
Delta = FFT_for_G(eta, urange)


plt.figure(figsize=(11, 6))

plt.plot(k0range, Delta, 'r', label=r'$\Delta^<_\bar{k}(k_0, \bar{\eta})$ FFT')

plt.title(r'$\Delta^<_\bar k(k_0, \bar \eta)$ as a function of $k_0$'
          r' when $\bar \eta={a},\ k={b}$'.format(a=eta, b=k), fontsize=14)
plt.xlabel(r'$k_0 \in \left[k + L/\bar\eta, k - L/\bar\eta \right],\ L={a}$'
           .format(a=L), fontsize=13)

plt.axvline(0, c='gray')  # comment out when |eta| >> 1 (out of range)
plt.axhline(0, c='gray')

plt.axvline(k, c='b', linestyle='--')
plt.plot(k, integral_k, 'bD',
         label=r'$\Delta^<_\bar{k}(k_0=k, \bar{\eta})$')

# benchmarking for FFT
for i in range(5):
    randk0 = random.choice(k0range)
    plt.plot(randk0, Delta_benchmark_integral(randk0, eta), 'go')

plt.legend(loc='upper left', fontsize=14, shadow=True)
plt.grid()
plt.show()


# TODO: vihkosta löytyy jutut TÄSTÄ ALASPÄIN ON TESTAILUA
# MUOKKAA KOODI LUETTAVAKSI

print('Deltan maksimikohta k_0 =', k0range[np.argmax(Delta)])
newk0range = k0range[0:2*np.argmax(Delta)]
newDelta = Delta[0:2*np.argmax(Delta)].real
M = sum(newDelta)*(newk0range[1] - newk0range[0])
print('N =', M)
omega_k = sum(newk0range*newDelta)*(newk0range[1] - newk0range[0])/M
print('omega_k =', omega_k)


plt.figure(figsize=(11, 6))

plt.plot(newk0range, newDelta, 'r',
         label=r'$\Delta^<_\bar{k}(k_0, \bar{\eta})$ FFT')

plt.title(r'Symmetric around max, $\Delta^<_\bar k(k_0, \bar \eta)$ '
          'as a function of $k_0$'
          r' when $\bar \eta={a},\ k={b}$'.format(a=eta, b=k), fontsize=14)
plt.xlabel(r'$k_0 \in \left[\max - \lambda, \max + \lambda \right]$',
           fontsize=13)

plt.axvline(0, c='gray')  # comment out when |eta| >> 1 (out of range)
plt.axhline(0, c='gray')

plt.axvline(k, c='b', linestyle='--')
plt.plot(k, integral_k, 'bD',
         label=r'$\Delta^<_\bar{k}(k_0=k, \bar{\eta})$')

plt.legend(loc='upper left', fontsize=14, shadow=True)
plt.grid()
plt.show()


def deltaDelta(eta_bar, D):
    Delta = FFT_for_G(eta_bar, urange)
    newk0range = k0range[0:2*np.argmax(Delta)]
    newDelta = Delta[0:2*np.argmax(Delta)].real
    M = sum(newDelta)*(newk0range[1] - newk0range[0])
    omega_k = sum(newk0range*newDelta)*(newk0range[1] - newk0range[0])/M
    leftlimitindex = np.argmin(abs(newk0range - (omega_k - D)))
    rightlimitindex = np.argmin(abs(newk0range - (omega_k + D)))
    k0range_cutright = newk0range[:rightlimitindex]
    k0range_cut = k0range_cutright[leftlimitindex:]
    Delta_cutright = newDelta[:rightlimitindex]
    Delta_cut = Delta_cutright[leftlimitindex:]
    return sum(Delta_cut)*(k0range[1] - k0range[0])/M



Drange = np.linspace(1, 500, num=100)

deltajono = np.empty(len(Drange))
for i in range(len(Drange)):
    deltajono[i] = deltaDelta(eta, Drange[i])


plt.plot(Drange, deltajono)
plt.grid()

# yes! funktio deltaDelta taitaa toimia
