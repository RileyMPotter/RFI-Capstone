When running every code but the one that implements Shapiro-Wilks, the output will be in the form of a time series. 

When running Shapiro-Wilks script, apply the following two-step approch. First run Shapiro_Wilks_for_Yechan.m followed by dedisperse_PSR_J1713_fast.m. 

Note that swtest.m and normcdf2.m are functions used by the Shapiro-Wilks script. Therefore, keep them in the same directory with the main script.    

Data must be placed in the same directory as the main scripts. 

Here is a list of methods to run: 

Spectral Kurtosis (SK) 
Median Absolute Deviation (MAD) 
Shapiro-Wilks (SW) 
Spectral Entropy (SE) 
Spectral Relative Entropy symmetric (SRE_s) 
Spectral Relative Entropy asymmetric (SRE_a)   