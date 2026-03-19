import numpy as np
np.random.seed(2) # for reproducibility

import numba as nb
from numba import cfunc, njit, prange, vectorize, float64

from NumbaQuadpack import quadpack_sig
from NumbaQuadpack import dqags as quad

#import scipy as sp

import astropy.units as u
from naima.models import TableModel, InverseCompton, Synchrotron

import warnings
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter('ignore', SyntaxWarning)

# Seed Photon Fields for Naima
seed_fields = [['CMB',  2.7*u.K, 0.260*u.eV/u.cm**3],
               ['IR',   2e1*u.K, 0.600*u.eV/u.cm**3],
               ['Star', 5e3*u.K, 0.600*u.eV/u.cm**3],
               ['UV',   2e4*u.K, 0.100*u.eV/u.cm**3]]

# Constants and Cross-Sections, units are in eV, cm, s
me2 = (0.51099895069 * 1e6)**2 # Rest mass of the electron squared in eV (0.51 MeV) ignoring 1/c^2
c = 2.998e10 # cm/s
h = 4.135667696e-15 # eV*s
hbar = h/(2*np.pi) # eV*s
mu0 = 1.256637e-6 # N*A^-2
sig_t = 6.65e-25 #cm^2
kb = 8.617e-5 # eV/K

@njit(cache=True)
def sig_ics(E_g, e_g, E_e): # Inputs in energies of eV
    '''
    E_g is the energy of the up-scattered photon
    E_e is the energy of the colliding electron
    e_g is the energy of the target photon
    '''
    z = E_g / E_e
    b = 4 * e_g * E_e / me2
    pre_fac = 3 * sig_t * me2 / (4 * e_g * E_e**2)
    first = 1
    second = z**2 / (2 * (1-z))
    third = z / (b * (1-z))
    fourth = 2 * z**2 / (b**2 * (1-z))
    fifth = z**3 / (2 * b * (1-z)**2)
    six1 = (2 * z / (b * (1-z)))
    six2 = np.log(b * (1-z) / z)
    sixth = six1 * six2
    return pre_fac * (first + second + third - fourth - fifth + sixth) # Outputs cm^2 eV^-1

@njit(cache=True)
def synch_loss(Ee, B):# Input in uG, eV, Equation 5.8 in Dan's textbook
    '''
    B is the magnetic field in micro Gauss
    Ee is the energy of the electron in eV
    '''
    return 2.28e-17 * 1e9 * (Ee*1e-9)**2 * (B/3)**2 # Output in eV/s

@njit(cache=True)
def IC_loss(Ee): # Input in eV, K
    '''
    Ee is the energy of the electron in eV
    Temp is the temperature of the photon field in K
    '''
    Temps = [2.7, 2e1, 5e3, 2e4]
    factors = [1.0362812027180732, 0.0007943110365197878, 2.0334362534907387e-13, 1.3238517275329904e-16]
    ICs = np.zeros(len(Temps))
    for n in prange(len(Temps)):
        Temp = Temps[n]
        gam_k2 = (3 * np.sqrt(5) / (8 * np.pi * Temp * kb))**2 * me2
        gam_e2 = Ee**2 / me2
        num = 32 * np.pi**5 * sig_t * kb**4 * Temp**4 * gam_e2 * gam_k2
        denom = 45 * c**2 * h**3 * (gam_e2 + gam_k2)
        ICs[n] = num * factors[n] / denom
    return np.sum(ICs) # Output in eV/s

@njit(cache=True)
def total_loss(Ee, B): # Input in eV, uG, K
    synch = synch_loss(B, Ee)
    IC = IC_loss(Ee)
    return synch + IC

@cfunc(quadpack_sig)
def dEe_dt(E, _data):
    data = nb.carray(_data, (2,))
    B = data[0]
    loss = total_loss(E, B)
    return -1 / loss 
dEe_dt_integrand = dEe_dt.address
@njit
def get_t(E0, Ee, B): # Input in eV, eV, uG, K
    '''
    E0 is the initial energy of the electron in eV
    Ee is the final energy of the electron in eV
    B is the magnetic field in micro Gauss
    Temps (array) is the temperatures of the background photon fields in K
    '''
    data = np.array([B])
    integral, _, _ = quad(dEe_dt_integrand, E0, Ee, data=data)
    return integral

@vectorize([float64(float64, float64, float64)],nopython=True)
def get_Ee(T, E0, B):
    '''
    t is the time in seconds
    E0 is the initial energy of the electron in eV
    B is the magnetic field in micro Gauss
    Temps (array) is the temperatures of the background photon fields in K
    '''
    Ees = np.logspace(np.log10(E0), 6, 300)
    times = np.zeros(len(Ees))
    for n, i in enumerate(Ees):
        t = get_t(E0, i, B)
        if n == 0:
            times[n] = t
        else:
            if t > np.nanmax(times):
                times[n] = t
            else:
                times[n] = np.nan
    Ee = np.interp([T], times, Ees)
    return Ee[0]

@cfunc(quadpack_sig)
def _ldiff(E, _data):
    B = _data[0]
    delta = 1/3
    D0 = 3.86e28
    top = D0 * (E / 1e9)**delta
    bottom = -1 * total_loss(E, B)
    return top / bottom
Ldiff_integrand = _ldiff.address
@njit
def Ldiff(E0, Ee, B): # Input units of eV, eV, uG, cm^2/s, dimensionless, from Arxiv paper Eq 2.7, currently only working for 1 photon field temperature
    '''
    E0 is the initial energy of the electron in eV
    Ee is the final energy of the electron in eV
    B is the magnetic field in micro Gauss
    D0 is the diffusion coefficient in ?
    delta is the ?
    Temps (array) is the temperatures of the background photon fields in K
    '''
    data = np.array([B])
    integral, _, _ = quad(Ldiff_integrand, E0, Ee, data=data)
    return np.sqrt(integral) # Output in cm

@njit
def electron_number_density(E0, Ee, r, B, A, Ec, Q0): # Inputs in eV, eV, s, cm, uG, Electron PL Norm(Q)/Index(A)/Cut-off(Ec), cm^2/s, dimensionless
    '''
    E0 is the initial energy of the electrons in eV
    t is the time in seconds
    B is the magnetic field in micro Gauss
    Q0 is the normization factor of the electron spectrum in ?
    A is the power law index of the electrons
    Ec is the cutoff energy of the electrons in eV
    Temps (array) is the temperatures of the background photon fields in K
    radius is the radius away from the source center to do the calculation in cm
    '''
    L = Ldiff(E0, Ee, B)
    _top = Q0 * E0**(2-A)
    _bottom = 8 * np.pi**(3/2) * Ee**2 * L**3
    _exp1 = np.exp(-1 * E0 / Ec)
    _exp2 = np.exp(-1 * r**2 / (4 * L**2))
    return _top / _bottom * _exp1 * _exp2

def tsyn(E0, B, Ef=None): # Input in eV, µG
    if Ef == None:
        Ef = E0/2
    E_diff = (-1/Ef) + (1/E0)
    bottom = 57 * (B/3)**2
    return -2.5e27 / bottom * E_diff

def get_trange(B, tmax): # Input in µG, yrs
    E0s = np.logspace(17, 12, 300)
    ts = []
    for E in E0s:
        _t = tsyn(E, B)
        ts.append(_t)
        if _t > (tmax * 3.154e7):
            return ts
    return ts

def get_binwidths(ts):
    bin_widths = np.zeros_like(ts)
    for i in range(len(ts)):
        if i == 0:
            #low = ts[0]
            low = ts[1] - np.average([ts[1], ts[0]])
        else:
            low = ts[i] - np.average([ts[i], ts[i-1]])
        if i == len(ts) - 1:
            high = low
        else:
            high = np.average([ts[i], ts[i+1]]) - ts[i]
        bin_widths[i] = high + low
    return bin_widths


class LeptonicModel:
    def __init__(self, B, Q0, A, Ec, t, verbose=False, D=5000):
        # Fittd parameters of the source
        self.B = B
        self.Q0 = Q0
        self.A = A
        self.Ec = Ec
        self.t = t
        self.verbose = verbose
        
        # Fixed parameters of the source
        self.D =  D * 3.086e18 # pc to cm
        self.Dpm = 100 * 3.086e18 # 100 pc in cm
        
        # Electrons
        self.E0s = np.logspace(12, 17, 300) # eV
        self.Ees = get_Ee(self.t, self.E0s, self.B)
        
    def gamma_spectrum_r(self, Egs, r, n_dist=0):
        '''
        Egs are the energy of the gamma ray in eV
        r is the distance from the source in cm
        t is the time in seconds
        B is the magnetic field in micro Gauss
        Q0 is the normization factor of the electron spectrum in ?
        A is the power law index of the electrons
        Ec is the cutoff energy of the electrons in eV
        D0 is the diffusion coefficient in cm^2/s
        delta is the index of diffusion (not used currently)
        Temps (array) is the temperatures of the background photon fields in K
        '''
        dNes = np.zeros(len(self.Ees))
        for m, Ee in enumerate(self.Ees):
            E0 = self.E0s[m]
            dNes[m] = electron_number_density(E0, Ee, r, self.B, self.A, self.Ec, self.Q0)
        try:
            table_model = TableModel(self.Ees[dNes > 0]*u.eV, dNes[dNes > 0]/u.eV, amplitude=1)
            IC_model = InverseCompton(table_model, seed_photon_fields=seed_fields, Eemin=np.min(Egs)*u.eV, Eemax=np.max(self.Ees)*u.eV)
            #Syn_model = Synchrotron(table_model, B=self.B * u.uG, , Eemin=np.min(self.Ees)*u.eV, Eemax=np.max(self.Ees)*u.e)
            flux_IC = IC_model.flux(Egs * u.eV, distance=n_dist*u.kpc)
            #flux_Syn = Syn_model.flux(Egs * u.eV, distance=n_dist*u.kpc)
        except:
            return np.zeros(len(Egs)) / u.eV / u.s
        return flux_IC # Output in 1/eV/s
    
    def gamma_spectrum_at_Egs_theta(self, Egs, theta, n_points=49):
        '''
        Egs are the energy of the gamma ray in eV
        theta is the angle from the source in radians
        '''
        log_range = np.logspace(6.27, np.log10(self.Dpm), int(n_points/2)) # cm, values found to cover best range
        D_range = self.D + np.concatenate((-1*np.flip(log_range), [0], log_range)) # cm
        _integrand = np.zeros((len(Egs), len(D_range)))
        for n, l in enumerate(D_range):
            r = np.sqrt(l**2 + self.D**2 - 2 * l * self.D * np.cos(theta))
            _fluxs = self.gamma_spectrum_r(Egs, r)
            for m, _flux in enumerate(_fluxs.value):
                if _flux > 0:
                    _integrand[m, n] = _flux # 1/eV/s
                else:
                    _integrand[m, n] = 0 # 1/eV/s
        # Integrate over the line of sight
        _integral = np.trapezoid(_integrand, D_range, axis=-1)
        return _integral # Output in 1/eV/s/cm2
    
    def gamma_spectrum_full_morph(self, Egs):
        '''
        Egs are the energy of the gamma ray in eV
        Returns the full gamma spectrum at all angles
        '''
        # Create a grid of angles
        fluxs = np.zeros((len(Egs), len(self.thetas))) # Initialize the fluxes array
        for n, theta in enumerate(self.thetas):
            fluxs[:, n] = self.gamma_spectrum_at_Egs_theta(Egs, theta)
        
        out_fluxs = np.zeros(len(Egs))
        for n, en in enumerate(fluxs):
            _integrand = 2 * np.pi * en * np.sin(self.thetas)
            out_fluxs[n] = np.trapezoid(_integrand, self.thetas)        
        return out_fluxs
    
    def gamma_spectrum_morph(self, Egs):
        '''
        Egs are the energy of the gamma ray in eV
        Returns the full gamma spectrum at all angles
        '''
        # Create a grid of angles
        fluxs = np.zeros((len(Egs), len(self.thetas))) # Initialize the fluxes array
        for n, theta in enumerate(self.thetas):
            fluxs[:, n] = self.gamma_spectrum_at_Egs_theta(Egs, theta)
        return fluxs # Output in 1/eV/cm^2/s/sr

    
    def __call__(self, Egs, thetas=None, get_flux=False, get_map=False):
        '''
        Egs are the energy of the gamma ray in eV
        thetas are the angles from the source in radians, if None, returns the full gamma spectrum
        Returns the flux of the gamma ray in erg/cm^2/s
        '''
        if len(thetas) > 0:
            self.thetas = thetas
        else:
            self.thetas = np.linspace(0, 0.5, 20) * np.pi / 180 # Twenty points from 0 to 0.5 degrees in radians
        if get_flux == True and thetas is not None:
            return self.gamma_spectrum_full_morph(Egs)
        elif get_map == True:
            return self.gamma_spectrum_morph(Egs)
        else:
            return Exception('No valid input for spectrum or map was given')
