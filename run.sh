#!/bin/bash

# Fit the spectrum
python fit_flux.py

# Get the fitted flux
python get_flux.py

# Generate the radial profile
python get_map.py

echo "Crab Fit is Done" | ssmtp -v curtisginsbe@wisc.edu
