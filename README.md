# Leptonic Modeling of PeVatrons
Leptonic modeling of VHE/UHE gamma rays

Currently this is run on a server with 64 cores, that is why the number of workers for the fitting is 60. It still takes hours to do a single fit, so be warned if you try to run this that it is slow.

Code for the paper: https://arxiv.org/abs/2603.17012

Current dependencies:
- NumPy (scipy=1.14 wants numpy <= 2.2)
- SciPy (get a multiprocessing error if using version >= 1.15)
- Numba
- Naima
- NumbaQuadpack (https://github.com/Nicholaswogan/NumbaQuadpack)
- AstroPy
- tqdm
- lmfit
