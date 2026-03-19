# leptonic_modelling
Leptonic modelling focused on microquasars

Currently this is run on a workstation with 64 cores, that is why the number of workers for the fitting is 60. It still takes hours to do a single fit, so be warned if you try to run this that it is slow.

Non-Naima parts are based on:
- DOI: https://doi.org/10.1103/PhysRevD.96.103013
- ISBN: 9780691235042

Current dependencies:
- NumPy (scipy=1.14 wants numpy <= 2.2)
- SciPy (get a multiprocessing error if using version >= 1.15)
- Numba
- Naima
- NumbaQuadpack (https://github.com/Nicholaswogan/NumbaQuadpack)
- AstroPy
- tqdm
- lmfit
