# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:49:02 2025

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
N = 4000

L = 250  # limits for k0 (when eta = -1)

# --------------


nu = np.sqrt(((d - 1)/2)**2 - (m/H)**2)

# integration variable u = Delta_eta/(2eta_bar) runs over (-1, 1)
epsilon = 1e-10  # small threshold in order to avoid limit points (nan)
urange = np.linspace(-1 + epsilon, 1 - epsilon, N)


def k0range(eta):
    """
    Range of k_0's for plotting [k - L/|eta|, k + L/|eta|]

    Parameters
    ----------
    eta : float
        mean of conformal times (eta + eta')/2

    Returns
    -------
    array-like
        range of k_0 for plotting

    """
    return np.linspace(-L/abs(eta) + k, L/abs(eta) + k, N)


def integrand_for_Delta(u, k_0, eta):
    """
    This integrated over u in (-1, 1) should be Delta^<_k

    Parameters
    ----------
    u : array-like
        Integration variable u = Deltaeta/etabar;
        Deltaeta = eta' - eta: conformal time difference
    k_0 : array-like
        frequency
    eta : array-like
        mean of conformal times 1/2(eta' + eta)

    Returns
    -------
    array-like
        Integrand for given values

    """
    hankel_a = sc.special.hankel1(nu, k*eta*(u - 1))
    hankel_b = sc.special.hankel2(nu, -k*eta*(u + 1))
    c = (1 - u**2)**(3/2)*np.exp(-1j*k_0*2*u*eta)
    return np.pi/2*H**2*eta**4*c*hankel_a*hankel_b


# Integral using scipy quad (only real part (imaginary part is zero))
def Delta_benchmark_integral(k_0, eta):
    """
    Uses scipy quad to evaluate the integral for Delta at given points

    Parameters
    ----------
    k_0 : float
        frequency
    eta : float
        mean of conformal times

    Returns
    -------
    float
        Delta at given points

    """
    return sc.integrate.quad(integrand_for_Delta, -1, 1,
                             args=(k_0, eta))[0]


def plot_integrand(k_0, eta):
    """
    Plots the integrand for Delta as function of integration variable u

    Parameters
    ----------
    k_0 : array-like
        frequency
    eta : float
        mean of conformal times 1/2(eta' + eta)

    Returns
    -------
    None.

    """
    # Plotting the integrand for fixed value k0
    integrand_k0 = integrand_for_Delta(k_0, eta, urange)
    plt.figure()
    plt.plot(urange, integrand_k0, 'r')  # plots the real part of I
    # plt.plot(urange, integrand_k0.imag, 'b')  # imaginary part of I
    plt.title('Integrand as a function of integration variable u')
    plt.xlabel(r'$u=\frac{\Delta \eta}{2\bar{\eta}}$')
    plt.ylabel(r'$I(u, \bar{\eta}, k_0)$')
    plt.show()


def grid_for_FFT(eta):
    """
    input matrix for FFT

    Parameters
    ----------
    eta : float
        mean of conformal times (eta + eta')/2

    Returns
    -------
    array-like
        NxN matrix for given eta_bar

    """
    # input matrix for FFT
    x1, y1 = np.meshgrid(urange, k0range(eta))
    return np.exp(-1j*x1*2*y1*eta)*(urange[1] - urange[0])


# FFT using the above defined grid
def FFT_for_G(eta, u, gridmatrix):
    """
    Fourier transform of G wrt. u, this is Delta as a function of k0

    Parameters
    ----------
    eta : array-like
        mean of conformal times 1/2(eta' + eta)
    u : array-like
        Integration variable u = Deltaeta/etabar;
        Deltaeta = eta' - eta: conformal time difference
    gridmatrix : array-like
        NxN matrix for the FFT, depends on etabar

    Returns
    -------
    array-like
        Fourier-transform for given values, Delta as a function of k0

    """
    hankel_a = sc.special.hankel1(nu, k*eta*(u - 1))
    hankel_b = sc.special.hankel2(nu, -k*eta*(u + 1))
    c = (1 - u**2)**(3/2)
    A = hankel_a*hankel_b*c
    return np.pi/2*H**4*eta**6*np.matmul(gridmatrix, A)
    # divided original Delta by a^2=1/(H**2*eta**2)


def deltaD(D, newk0range, newDelta, omega_k):
    """
    Gives the quantity delta_Delta

    Parameters
    ----------
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
    # k0range_cut = k0range_cutright[leftlimitindex:]  # for plotting
    Delta_cutright = newDelta[:rightlimitindex]
    Delta_cut = Delta_cutright[leftlimitindex:]
    return sum(Delta_cut)*(newk0range[1] - newk0range[0])


def plotDelta(k0rangej, Delta, eta, a):
    """
    Plots Delta as a function of k0 for given value of eta

    Parameters
    ----------
    k0rangej : array-like
        plotting range
    Delta : array-like
        Delta values
    eta : float
        mean of conformal times
    a : float
        scale factor

    Returns
    -------
    None.

    """
    constantstr = '\n'.join((
            r'$\bar\eta={eta}$'.format(eta=eta),
            r'$k={k}$'.format(k=k),
            r'$a={a}$'.format(a=a),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.figure(figsize=(11, 6))
    plt.plot(k0rangej, Delta, 'r',
             label=r'$\Delta^<_\bar{k}(k_0, \bar{\eta})$ FFT')
    plt.title(r'$\Delta^<_\bar k(k_0, \bar \eta)$'
              r' as a function of $k_0$', fontsize=14)
    plt.xlabel(r'$k_0 \in \left[k+L/\bar\eta,k-L/\bar\eta \right]$, '
               '$L={L}$'.format(L=L), fontsize=13)
    # axes
    plt.axvline(0, c='gray')  # comment when |eta| >> 1 (out of range)
    plt.axhline(0, c='gray')

    plt.axvline(k, c='b', linestyle='--')
    plt.plot(k, Delta_benchmark_integral(k, eta), 'bD',
             label=r'$\Delta^<_\bar{k}(k_0=k, \bar{\eta})$')

    # testing for FFT
    # for i in range(5):
    #     randk0 = random.choice(k0rangej)
    #     plt.plot(randk0, Delta_benchmark_integral(randk0, eta),
    #              'go')

    plt.grid()
    plt.legend(loc='upper left', fontsize=14, shadow=True)
    plt.text(0.75*k0rangej[-1], 0.65*max(Delta),
             s=constantstr, fontsize=14, bbox=constantbox)
    plt.show()


def plotdeltaD(eta, cutk0, norm, omega_k, a, cutDelta):
    """
    Plots deltaD as a function of D in order to check close Delta behaves
    to Dirac delta at a given eta

    Parameters
    ----------
    eta : float
        mean of conformal times
    cutk0 : array-like
        k0range for given eta cut to be symmetric around max of Delta
    norm : float
        integral of Delta over whole k0 range (same as rho0)
    omega_k : float
        integral of k0*Delta over whole k0 range (same as rho1/rho0)
    a : float
        scale factor
    cutDelta : array-like
        Deltarange for given eta cut to be symmetric around max of Delta

    Returns
    -------
    None.

    """
    Drange = np.linspace(0, cutk0[-1], num=1000)
    deltaseq = np.empty(len(Drange))
    for i in range(len(Drange)):
        deltaseq[i] = deltaD(Drange[i], cutk0, cutDelta, omega_k)
    constantstr = '\n'.join((
            r'$\bar\eta={eta}$'.format(eta=eta),
            r'$k={k}$'.format(k=k),
            r'$a={a}$'.format(a=a),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.figure(figsize=(11, 6))

    plt.plot(Drange, deltaseq, c='g')  # for normed one, deltaseq/norm

    plt.ylabel(r'$\delta_\Delta$', fontsize=13)
    plt.xlabel(r'$\Delta$', fontsize=13)
    plt.title(r'$\delta_\Delta$', fontsize=14)
    plt.text(0.8*Drange[-1], 0.2*max(deltaseq),
             s=constantstr, fontsize=14,
             bbox=constantbox)

    plt.grid()
    plt.show()


def plot_omega_k(omega_ks, spikelocs, etas):
    """
    Plots omega_ks and max points of Delta as a function of eta
    (omega_k is not the same as in f functions)

    Parameters
    ----------
    omega_ks : array-like
        integral of k0*delta normed (same as rho1/rho0)
    spikelocs : array-like
        locations of max of Delta
    etas : array-like
        eta values for plotting axis

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 6))
    plt.plot(etas, omega_ks, 'b', label=r'$\omega_k$')
    plt.plot(etas, spikelocs, 'r',
             label=r'location of the spike $\bar\omega$')
    plt.title(r'$\omega_k$ vs $\bar\omega$', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    plt.ylabel(r'units of $k$', fontsize=13)
    plt.legend(loc='lower left', fontsize=14)
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[0], 0.6*omega_ks[0],
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.grid()
    plt.show()


def plot_rho0_and_drho0(rho0s, drho0s, etas):
    """
    Plots rho0 and its eta derivative as functions of eta

    Parameters
    ----------
    rho0s : array-like
        integrals of Delta over k0
    drho0s : array-like
        eta derivatives of rho0
    etas : array-like
        eta values for plotting

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 6))
    plt.plot(etas, rho0s, 'b')
    plt.title(r'$\rho_{0k}(\bar\eta)$', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[-1], 0.7*max(rho0s),
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.grid()
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.plot(etas, drho0s, 'r')
    plt.title(r'$\partial_\eta\rho_{0k}(\bar\eta)$', fontsize=14)
    plt.xlabel(r'$\bar\eta$')
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[-1], 0.7*min(drho0s),
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.grid()
    plt.show()


def plot_rho2(rho2s, etas):
    """
    Plots rho2 as a function of eta

    Parameters
    ----------
    rho2s : array-like
        integral of k0**2*Delta
    etas : array-like
        eta values for plotting

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 6))
    plt.plot(etas, rho2s, 'b')
    plt.title(r'$\rho_{2k}(\eta)$', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[0], 0.7*max(rho2s),
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.grid()
    plt.show()


def plot_f(rho0s, drho0s, rho2s, etas):
    """
    Plots f+ and f- as a function of eta

    Parameters
    ----------
    rho0s : array-like
        integrals of Delta wrt. k0
    drho0s : array-like
        eta derivative of rho0
    rho2s : array-like
        rho2 values, integral of k0**2*Delta
    etas : array-like
        eta values for plotting

    Returns
    -------
    None.

    """
    a2_list = 1/(etas*H)**2  # scale factor a squared
    m_eff2 = a2_list*(m**2 - 2*H**2)  # efective mass squared
    omega_k = np.emath.sqrt(k**2 + m_eff2)
    fminus = omega_k*rho0s - rho2s/omega_k + 1j*etas*H*drho0s/2
    fplus = omega_k*rho0s - rho2s/omega_k - 1j*etas*H*drho0s/2

    plt.figure(figsize=(11, 6))
    plt.plot(etas, fminus.real, 'g')
    plt.title(r'$f_{\bar{k}c}^\pm(\bar\eta)$ real part', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[0], 0.7*min(fminus.real),
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.grid()
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.plot(etas, fminus.imag, 'b', label=r'$\Im(f_{\bar{k}c}^-)$')
    plt.title(r'$f_{\bar{k}c}^\pm(\bar\eta)$ imaginary part', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[0], 0,
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.legend(loc='upper right', fontsize=14, shadow=True)
    plt.grid()
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.plot(etas, fplus.imag, 'r', label=r'$\Im(f_{\bar{k}c}^+)$')
    plt.title(r'$f_{\bar{k}c}^\pm(\bar\eta)$ imaginary part', fontsize=14)
    plt.xlabel(r'$\bar\eta$', fontsize=13)
    constantstr = '\n'.join((
            r'$k={k}$'.format(k=k),
            r'$H={H}$'.format(H=H),
            r'$m={m}$'.format(m=m)))
    constantbox = dict(boxstyle='round',
                       facecolor='turquoise', alpha=0.5)
    plt.text(etas[0], 0,
             s=constantstr, fontsize=14,
             bbox=constantbox)  # textbox location depends on the parameters
    plt.legend(loc='upper right', fontsize=14, shadow=True)
    plt.grid()
    plt.show()


def plotting(etas, Deltaplot=False, deltaDplot=False, rho0plot=False,
             rho2plot=False, omega_kplot=False, fplot=False):
    """
    Select what you want to plot for given array of etas

    Parameters
    ----------
    etas : array-like
        means of conformal times for plotting
    Deltaplot : Boolean, optional
        True plots Delta for every given eta, when a lot of etas select False.
        The default is False.
    deltaDplot : Boolean, optional
        True plots deltaD for every given eta, when a lot of etas select False.
        The default is False.
    rho0plot : Boolean, optional
        plots rho0 and its eta derivative as function of etas.
        The default is False.
    rho2plot : Boolean, optional
        plots rho2 as function of etas. The default is False.
    omega_kplot : Boolean, optional
        Plots omega_k(not the same as in f) and max of Delta in same figure as
        a function of etas. The default is False.
    fplot : Boolean, optional
        Plots f+ and f- as functions of etas. The default is False.

    Returns
    -------
    None.

    """
    etanum = len(etas)  # number of etas
    rho0s = np.empty(etanum)
    rho2s = np.empty(etanum)
    omega_ks = np.empty(etanum)
    spikelocs = np.empty(etanum)
    for j in range(len(etas)):
        a = -1/(etas[j]*H)
        Delta = FFT_for_G(etas[j], urange, grid_for_FFT(etas[j])).real
        k0rangej = k0range(etas[j])
        if Deltaplot:
            plotDelta(k0rangej, Delta, etas[j], a)
        maxindex = np.argmax(Delta)
        cutDelta = Delta[:2*maxindex]
        cutk0 = k0rangej[:2*maxindex]
        spikelocs[j] = k0rangej[maxindex]
        if (rho0plot or deltaDplot or fplot or omega_kplot):
            rho0s[j] = sum(cutDelta)*(cutk0[1] - cutk0[0])
        if (deltaDplot or omega_kplot):
            omega_k = sum(cutk0*cutDelta)*(cutk0[1] - cutk0[0])/rho0s[j]
            omega_ks[j] = omega_k
            if deltaDplot:
                plotdeltaD(etas[j], cutk0, rho0s[j], omega_k, a, cutDelta)
        if (rho2plot or fplot):
            rho2s[j] = sum(cutk0**2*cutDelta)*(cutk0[1] - cutk0[0])
    if (rho0plot or deltaDplot or fplot):
        drho0s = np.gradient(rho0s, etas)
    if omega_kplot:
        plot_omega_k(omega_ks, spikelocs, etas)
    if rho0plot:
        plot_rho0_and_drho0(rho0s, drho0s, etas)
    if rho2plot:
        plot_rho2(rho2s, etas)
    if fplot:
        plot_f(rho0s, drho0s, rho2s, etas)


# input etabar values
# etarange = [-0.1, -0.5, -1, -4]
etarange = np.linspace(-5, -0.1, num=50)
# (etabar) mean of conformal times (eta' + eta)/2 in range (-inf, 0)
# -k*eta > 1 subhorizon
# -k*eta < 1 superhorizon

# etas must be in order when plotting rhos or fs
# NOTE!: when number of etas is large, use: Deltaplot=False, deltaDplot=False
plotting(etarange, Deltaplot=False, deltaDplot=False, omega_kplot=False,
         rho0plot=True, fplot=True, rho2plot=True)

# Running time is large for dense etarange
