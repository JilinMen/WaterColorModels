# chlorophyll-a concentration remote sensing algorithms for Landsat 8/9
# By jmen@ua.edu
# 3/18/2025

import numpy as np

def chl_OC3(B1, B2, B3):
    """
    O'Reilly, J., Werdell, P., 2019. Chlorophyll algorithms for ocean color sensors - OC4, OC5
    & OC6. Remote Sens. Environ. 229, 32–47.
    B1-443
    B2-482
    B3-561
    """
    # chlulate maximum band ratio
    numerator = max(B1, B2)
    denominator = B3
    MBR = numerator / denominator

    # chlulate log10 of band ratio
    R = np.log10(MBR)
    
    # Coefficients
    c0 = 0.30963
    c1 = -2.40052
    c2 = 1.28932
    c3 = 0.52802
    c4 = -1.33825
    
    # chlulate chlorophyll-a concentration
    log_chla = c0 + c1*R + c2*R**2 + c3*R**3 + c4*R**4
    chla = 10**log_chla
    
    return chla

def chl_OCI_Hu2019(B1, B2, B3, B4):
    """
    OCI-Hu algorithm for estimating chlorophyll-a concentration (Hu et al., 2011)
    Hu, C., Lee, Z., Franz, B., 2011. Chlorophyll a algorithms for oligotrophic oceans: a novel
    approach based on three-band reflectance difference. J. Geophys. Res. Oceans 117
    (C1), C01011. https://doi.org/10.1029/2011JC007395.
    """
    # chlulate CI
    CI = B3 - (B1 + 0.5 * (B4 -  B1))

    Chl_ci = 10**(-0.4287 + 230.47 * CI)
    Chl_oc3 = chl_OC3(B1, B2, B3)
    
    Chl_L = 0.25
    Chl_H = 0.4
    Chl_hybird = Chl_oc3*(Chl_ci - Chl_L)/(Chl_H - Chl_L) + Chl_ci*(Chl_H - Chl_ci)/(Chl_H - Chl_L)
    
    Chl_final = Chl_hybird.copy()
    Chl_final[Chl_ci <= Chl_L] = Chl_ci[Chl_ci <= Chl_L]
    Chl_final[Chl_ci >= Chl_H] = Chl_oc3[Chl_ci >= Chl_H]    

    return Chl_final

def chl_Tassan(Rrs_412, Rrs_443, Rrs_490, Rrs_555):
    """
    Tassan algorithm for estimating chlorophyll-a concentration (Tassan, 1994)
    Tassan, S., 1994. Local algorithms using SeaWiFS data for the retrieval of phytoplankton,
    pigments, suspended sediment, and yellow substance in coastal waters. Appl. Opt. 33,
    2369–2378.
    Inputs:
    -------
    Rrs_412 : float
        Remote sensing reflectance at 412 nm in sr^-1
    Rrs_443 : float
        Remote sensing reflectance at 443 nm in sr^-1
    Rrs_490 : float
        Remote sensing reflectance at 490 nm in sr^-1
    Rrs_555 : float
        Remote sensing reflectance at 555 nm in sr^-1
        
    Output:
    -------
    chla : float
        Estimated chlorophyll-a concentration in mg/m^3
    """
    # chlulate X parameter
    X = (Rrs_443 / Rrs_555) * (Rrs_412 / Rrs_490)**3
    
    # Coefficients
    c0 = 0.51881
    c1 = -2.3535
    c2 = -0.027598
    c3 = -1.5
    
    # chlulate chlorophyll-a concentration
    log_chla = c0 + c1 * np.log10(X) + c2 * (np.log10(X))**2 + c3 * (np.log10(X))**3
    chla = 10**log_chla
    
    return chla

def ratio(Rrs_412, Rrs_443, Rrs_510, Rrs_555):
    """
    Tang algorithm for estimating chlorophyll-a concentration (Tang et al., 2004)
    Tang, J., Wang, X., Song, Q., Li, T., Chen, J., Huang, H., Ren, J., 2004. The statistic
    inversion algorithms of water constituents for the Huanghai Sea and the East China
    Sea. Acta Oceanologica Sinica 23 (4), 617–626.
    Inputs:
    -------
    Rrs_412 : float
        Remote sensing reflectance at 412 nm in sr^-1
    Rrs_443 : float
        Remote sensing reflectance at 443 nm in sr^-1
    Rrs_510 : float
        Remote sensing reflectance at 510 nm in sr^-1
    Rrs_555 : float
        Remote sensing reflectance at 555 nm in sr^-1
        
    Output:
    -------
    chla : float
        Estimated chlorophyll-a concentration in mg/m^3
    """
    # chlulate X parameter
    X = (Rrs_443 / Rrs_555) * (Rrs_412 / Rrs_510)**3
    
    # Coefficients
    c0 = -0.37457
    c1 = -3.7278
    c2 = -3.0679
    c3 = -1.0
    
    # chlulate chlorophyll-a concentration
    log_chla = c0 + c1 * np.log10(X) + c2 * (np.log10(X))**2 + c3 * (np.log10(X))**3
    chla = 10**log_chla
    
    return chla

def chl_Sun(Rrs_412, Rrs_443, Rrs_490, Rrs_555):
    """
    Sun algorithm for estimating chlorophyll-a concentration (Sun et al., 2010)
    Sun, L., Guo, M., Wang, X., 2010. Ocean color products retrieval and validation around
    China coast with MODIS. Acta Oceanologica Sinica 29, 21–27.
    Inputs:
    -------
    Rrs_412 : float
        Remote sensing reflectance at 412 nm in sr^-1
    Rrs_443 : float
        Remote sensing reflectance at 443 nm in sr^-1
    Rrs_490 : float
        Remote sensing reflectance at 490 nm in sr^-1
    Rrs_555 : float
        Remote sensing reflectance at 555 nm in sr^-1
        
    Output:
    -------
    chla : float
        Estimated chlorophyll-a concentration in mg/m^3
    """
    # chlulate X parameter
    X = (Rrs_443 / Rrs_555) * (Rrs_412 / Rrs_490)**3
    
    # Coefficients
    c0 = 0.118445
    c1 = -3.057611
    c2 = 3.0986261
    c3 = -0.8
    
    # chlulate chlorophyll-a concentration
    log_chla = c0 + c1 * np.log10(X) + c2 * (np.log10(X))**2 + c3 * (np.log10(X))**3
    chla = 10**log_chla
    
    return chla

def chl_OCT(Rrs_443, Rrs_490, Rrs_550, Rrs_555):
    """
    OCT algorithm for estimating chlorophyll-a concentration (Kim et al., 2016)
    Kim, W., Moon, J., Park, Y., Ishizaka, J., 2016. Evaluation of chlorophyll retrievals from
    Geostationary Ocean Color Imager (GOCI) for the North-East Asian region. Remote
    Sens. Environ. 184, 482–495.
    Inputs:
    -------
    Rrs_443 : float
        Remote sensing reflectance at 443 nm in sr^-1
    Rrs_490 : float
        Remote sensing reflectance at 490 nm in sr^-1
    Rrs_550 : float
        Remote sensing reflectance at 550 nm in sr^-1
    Rrs_555 : float
        Remote sensing reflectance at 555 nm in sr^-1
        
    Output:
    -------
    chla : float
        Estimated chlorophyll-a concentration in mg/m^3
    """
    # chlulate maximum band ratio
    numerator = max(Rrs_443, Rrs_490)
    denominator = Rrs_550
    MBR = numerator / denominator
    
    # Coefficients
    c0 = 0.415
    c1 = -29.056
    c2 = -3.312
    c3 = -83.980
    
    # chlulate a0 and a1
    a0 = c0 + c1 * Rrs_555
    a1 = c2 + c3 * Rrs_555
    
    # chlulate chlorophyll-a concentration based on Rrs(555) value
    if Rrs_555 < 0.003:
        # Use OCT algorithm for clear waters
        log_chla_oct = a0 + a1 * np.log10(MBR)
        chla = 10**log_chla_oct
    else:
        # Use OC3 algorithm for turbid waters (simplified here)
        R = np.log10(max(Rrs_443, Rrs_490) / Rrs_550)
        chla = 10**(0.2424 - 2.5828*R + 1.7057*R**2 - 0.3415*R**3 - 0.8818*R**4)
    
    return chla

def chl_Gitelson(Rrs_665, Rrs_715, Rrs_750):
    """
    Gitelson algorithm for estimating chlorophyll-a concentration (Gitelson et al., 2008)
    Gitelson, A., Dall'olmo, G., Moses, W., Rundquist, D., Barrow, T., Fisher, T., Gurlin, D.,
    Holz, J., 2008. A simple semi-analytical model for remote estimation of chlorophyll-a
    in turbid waters: Validation. Remote Sens. Environ. 112, 3582–3593.
    Inputs:
    -------
    Rrs_665 : float
        Remote sensing reflectance at 665 nm in sr^-1
    Rrs_715 : float
        Remote sensing reflectance at 715 nm in sr^-1
    Rrs_750 : float
        Remote sensing reflectance at 750 nm in sr^-1
        
    Output:
    -------
    chla : float
        Estimated chlorophyll-a concentration in mg/m^3
    """
    # chlulate term based on band ratios
    term = (1/Rrs_665 - 1/Rrs_715) * Rrs_750
    
    # Coefficients
    c0 = 23.09
    c1 = 117.42
    
    # chlulate chlorophyll-a concentration
    chla = c0 + c1 * term
    
    return chla
