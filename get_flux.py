import numpy as np
import sys
sys.path.append('./')
from leptonic_class_naima import *
import pickle
import warnings
from astropy.table import Table
warnings.simplefilter("ignore", UserWarning)


def get_flux(vars, B, xdata, t_max):
    Egs = np.logspace(np.log10(xdata[0]) - 1, np.log10(xdata[-1]) + 1, 60)
    Q0 = 10**vars[0]
    A = vars[1]
    Ec = 10**vars[2]
    
    ts = get_trange(B, t_max)
    bin_widths = get_binwidths(ts)
    fluxs = np.zeros((len(ts), len(Egs)))
    thetas = np.linspace(0, 0.5, 20) # degrees
    thetas = thetas * np.pi / 180  # convert to radians
    for n, t in enumerate(ts):
        Model = LeptonicModel(B, Q0, A, Ec, t, D=2000)
        _fluxs = Model(Egs, thetas=thetas, get_flux=True)
        fluxs[n] = _fluxs * bin_widths[n] / (t_max * 3.154e7)
    ind_1TeV = np.where(xdata >= 1e12)[0][0]
    ind_10TeV = np.where(xdata >= 10e12)[0][0]
    for n in range(len(fluxs)):
        if n != 0 and n != len(fluxs)-1:
            a = fluxs[n-1]
            b = fluxs[n]
            c = fluxs[n+1]
            if (b[ind_10TeV] > a[ind_10TeV] and b[ind_10TeV] > c[ind_10TeV]) or (b[ind_1TeV] > a[ind_1TeV] and b[ind_1TeV] > c[ind_1TeV]):
                fluxs[n] = (a + c) / 2
                    
    return fluxs # 1/eV/cm2/s

# Gamma-Ray Data
VTS_data = Table.read("VERITAS_2015_ICRC_Crab.ecsv", format="ecsv") # https://doi.org/10.22323/1.236.0792
xdata = VTS_data["e_ref"] * 1e12 # eV
ydata = VTS_data["dnde"] / 1e12 # eV-1 cm-2 s-1
sigma = VTS_data["dnde_err"] / 1e12 # eV-1 cm-2 s-1

B = 125 #uG
t_max = 1e3 # 1 kyr

with open(f'fit_Crab_B{B}.pkl', 'rb') as file:
    _fit = pickle.load(file)
ts = get_trange(B, t_max)
binwidths = get_binwidths(ts)
Q0 = 80 #_fit.params['Q0'].value
A = _fit.params['A'].value
Ec = _fit.params['Ec'].value

vars = (Q0, A, Ec)
print('Getting flux with:', vars)
fluxs = get_flux(vars, B, xdata, t_max)

np.save(f'flux_Crab_A{A*10:.0f}_Ec{Ec*10:.0f}_B{B:.0f}.npy', fluxs) # 1/eV/cm2/s
