#!/bin/bash

# Change to the proper directory
cd /home/curtisginsbe/leptonic_modelling_git

# Fit the spectrum
python fit_flux.py

# Get the fitted flux
python get_flux.py

# Generate the radial profile
python get_map.py

echo "Crab Fit is Done" | ssmtp -v curtisginsbe@wisc.edu
