import numpy as np
import lmfit
import sys
sys.path.append('./')
from leptonic_class_naima import *
import pickle
import warnings
from astropy.table import Table
warnings.simplefilter("ignore", UserWarning)


def fit_flux(vars, B, xdata, ydata, sigma, t_max): #t_max input in years
    Q0 = 10**80 #vars['Q0'].value
    A = vars['A'].value
    Ec = 10**vars['Ec'].value
    
    ts = get_trange(B, t_max)
    bin_widths = get_binwidths(ts)
    fluxs = np.zeros((len(ts), len(xdata)))
    thetas = np.linspace(0, 0.5, 20) # degrees
    thetas = thetas * np.pi / 180  # convert to radians
    for n, t in enumerate(ts):
        Model = LeptonicModel(B, Q0, A, Ec, t, D=2000)
        _fluxs = Model(xdata, thetas=thetas, get_flux=True)
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
    model = np.nansum(fluxs.transpose(), axis=1)
    model /= model[ind_1TeV]
    ydata /= ydata[ind_1TeV]
    resids = np.zeros(len(model))
    for n, sigma in enumerate(sigma):
        resids[n] = (ydata[n] - model[n]) / sigma
    return resids

# Gamma-Ray Data
VTS_data = Table.read("VERITAS_2015_ICRC_Crab.ecsv", format="ecsv") # https://doi.org/10.22323/1.236.0792
xdata = VTS_data["e_ref"] * 1e12 # eV
ydata = VTS_data["dnde"] / 1e12 # eV-1 cm-2 s-1
sigma = VTS_data["dnde_err"] / 1e12 # eV-1 cm-2 s-1

B = 125 # uG
t_max = 1e3 # 1 kyr

params = lmfit.Parameters()
#params.add('Q0', 85.0, min=40.0, max=100.0)
params.add('A', 1.5, min=1.0, max=3.0)
params.add('Ec', 14.5, min=13.0, max=16.0)

print(f'Fitting with B={B}')

fit = lmfit.minimize(fit_flux, params, args=(B, xdata, ydata, sigma, t_max), nan_policy='omit',
        method='differential_evolution', seed=2, disp=True, workers=50, polish=False)

with open(f'fit_Crab_B{B:.0f}.pkl', 'wb') as file:
   pickle.dump(fit, file)

lmfit.fit_report(fit)
print()
