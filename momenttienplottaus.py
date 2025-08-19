# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:15:02 2025

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


def rho_0k(eta_bar, k):
    """
    Gives the moment rho_0k at given values

    Parameters
    ----------
    eta_bar : array-like
        mean conformal time 1/2(eta' + eta)
    k : array-like
        norm of k_bar

    Returns
    -------
    array-like
        moment rho_0k at given points

    """
    hankel = abs(sc.special.hankel1(nu, -k*eta_bar))**2
    c = np.pi/2*H**2*eta_bar**4
    return c*hankel


# mathematican derivoima (ja sievennetty)
def rho_2k(eta_bar, k):
    hankel_a = (3/4 - (k*eta_bar)**2)*abs(sc.special.hankel1(nu, -k*eta_bar))**2
    hankel_b = -1/2*(k*eta_bar)**2*(abs(sc.special.hankel1(nu-1, -k*eta_bar))**2 + abs(sc.special.hankel1(nu+1, -k*eta_bar))**2)
    hankel_c = -3*(k*eta_bar)*sc.special.hankel2(nu, -k*eta_bar)*sc.special.hankel1(nu-1, -k*eta_bar)
    hankel_d = -3*(k*eta_bar)*sc.special.hankel1(nu, -k*eta_bar)*sc.special.hankel2(nu+1, -k*eta_bar)
    hankel_e = 1/2*(k*eta_bar)**2*sc.special.hankel2(nu, -k*eta_bar)*sc.special.hankel1(nu-2, -k*eta_bar)
    hankel_f = 1/2*(k*eta_bar)**2*sc.special.hankel2(nu, -k*eta_bar)*sc.special.hankel1(nu+2, -k*eta_bar)
    hankel_g = (k*eta_bar)**2*sc.special.hankel1(nu-1, -k*eta_bar)*sc.special.hankel2(nu+1, -k*eta_bar)
    hankel_h = hankel_c + hankel_d + hankel_e + hankel_f + hankel_g
    hankelit = hankel_a + hankel_b + hankel_h.real
    return -np.pi/8*(H*eta_bar)**2*hankelit


# numerical differentiation for benchmarking
def Drho_0kn(eta_bar, k):
    return np.gradient(rho_0k(eta_bar, k), eta_bar)


# not the same as numerical??
def Drho_0k(eta_bar, k):
    hankel_a = 2*abs(sc.special.hankel1(nu, -k*eta_bar))**2
    hankel_b = 1/2*k*eta_bar*((sc.special.hankel1(nu+1, -k*eta_bar)-sc.special.hankel1(nu-1, -k*eta_bar))*sc.special.hankel2(nu, -k*eta_bar)).real
    return np.pi*H**2*eta_bar**3*(hankel_a + hankel_b)


def rho_1k(eta_bar, k):
    a = -3/2*abs(sc.special.hankel1(nu, -k*eta_bar))**2
    b = k*eta_bar*((sc.special.hankel1(nu-1, -k*eta_bar)-sc.special.hankel1(nu+1, -k*eta_bar))*sc.special.hankel2(nu, -k*eta_bar)).imag
    return 1j*H**2*eta_bar**3*(a + b)


def fplus(eta_bar, k):
    omega = np.sqrt(k**2 + m**2)
    return omega*rho_0k(eta_bar, k) - rho_2k(eta_bar, k)/omega + 1j/2*Drho_0kn(eta_bar, k)


N = 300
etabar = np.linspace(-5, 0, N)
knorm = np.linspace(0, 40, N)
knormi = 5


rho0k = rho_0k(etabar, knormi)
Deta_rho0k = Drho_0k(etabar, knormi)
Deta_rho0kn = Drho_0kn(etabar, knormi)
rho1k = rho_1k(etabar, knormi)
rho2k = rho_2k(etabar, knormi)

f = fplus(etabar, knormi)


plt.figure()
plt.plot(etabar, rho0k, 'g')
plt.title(r'$\rho_{0k}(\bar{\eta})$')
plt.show()

plt.figure()
plt.plot(etabar, Deta_rho0k, 'b')
plt.plot(etabar, Deta_rho0kn, 'r')
plt.title(r'$\partial_\bar{\eta}\rho_{0k}(\bar{\eta})$')
plt.show()

plt.figure()
plt.plot(etabar, rho1k.imag, 'y')
plt.title(r'$\rho_{1k}(\bar{\eta})$')
plt.show()

plt.figure()
plt.plot(etabar, rho2k, 'r')
plt.title(r'$\rho_{2k}(\bar{\eta})$')
plt.show()

plt.figure()
plt.plot(etabar, f, 'm')
plt.title(r'$f^+(\bar{\eta})$')
plt.show()


# x1, y1 = np.meshgrid(etabar, knorm)
# ff = fplus(x1, y1)

# plt.figure()
# plt.contourf(x1, y1, ff, cmap='jet')
# plt.colorbar()
# plt.show()
