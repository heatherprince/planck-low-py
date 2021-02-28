import numpy as np
from scipy.stats import lognorm

nbins_EE=3
nbins_TT=2

#EE
lmin_list_EE, lmax_list_EE, mu_LN_EE, sig_LN_EE, loc_LN_EE=np.loadtxt(
        'data/lognormal_fit_'+str(nbins_EE)+'bins_EE.txt', unpack=True)
lmin_list_EE=lmin_list_EE.astype('int')
lmax_list_EE=lmax_list_EE.astype('int')

#TT
lmin_list_TT, lmax_list_TT, mu_LN_TT, sig_LN_TT=np.loadtxt(
        'data/lognormal_fit_'+str(nbins_TT)+'bins_TT.txt', unpack=True)
lmin_list_TT=lmin_list_TT.astype('int')
lmax_list_TT=lmax_list_TT.astype('int')



# binning
def get_binned_D_from_theory_Cls(ell, Cl, lmin_list, lmax_list):
    # convert from C to D=l(l+1)C_l/2pi, average in bin, convert back to C
    # loops through lmin and lmax
    # ell 2-29

    D_fac=ell*(ell+1)/(2*np.pi)
    Dl=D_fac*Cl

    Dl_bin=np.zeros(len(lmin_list))

    for i, lmin in enumerate(lmin_list):
        lmax = lmax_list[i]
        Dl_bin[i]=np.mean(Dl[lmin-2:lmax-2+1])
    return Dl_bin

def lognormal(x, mu, sig, loc=0):
    if x-loc==0:
        return 0
    LN=1/((x-loc)*sig*np.sqrt(2*np.pi))*np.exp(-(np.log(x-loc)-mu)**2/(2*sig**2))
    return LN

"""
EE
computes the log-likelihood of the Planck low-ell E mode polarization
using 3 log-normal bins
"""
def planck_lowE_binned_loglike(Cl_theory):
    # Cl_theory is a numpy array of Cl_EE from ell=2-30
    ell = np.arange(2, 30)
    Dl_theory_bin=get_binned_D_from_theory_Cls(ell , Cl_theory, lmin_list_EE, lmax_list_EE)
    loglike=0
    for i, D in enumerate(Dl_theory_bin):
        like=lognormal(D, mu_LN_EE[i], sig_LN_EE[i], loc_LN_EE[i])
        loglike+=np.log(like)
    return loglike

def planck_lowE_binned_loglike_cobaya(_theory={'Cl': {'ee': 29}}):
    ell = _theory.get_Cl(ell_factor=False)['ell'][2:30]
    Cl_theory = _theory.get_Cl(ell_factor=False)['ee'][2:30]

    Dl_theory_bin=get_binned_D_from_theory_Cls(ell , Cl_theory, lmin_list_EE, lmax_list_EE)

    loglike=0
    for i, D in enumerate(Dl_theory_bin):
        like=lognormal(D, mu_LN_EE[i], sig_LN_EE[i], loc_LN_EE[i])
        loglike+=np.log(like)
    return loglike

"""
TT
computes the log-likelihood of the Planck low-ell E mode polarization
using 2 log-normal bins
"""
def planck_lowT_binned_loglike(Cl_theory):
    # Cl_theory is a numpy array of Cl_TT from ell=2-30
    ell = np.arange(2, 30)
    Dl_theory_bin=get_binned_D_from_theory_Cls(ell , Cl_theory, lmin_list_TT, lmax_list_TT)
    loglike=0
    for i, D in enumerate(Dl_theory_bin):
        mu=mu_LN_TT[i]
        sigma=sig_LN_TT[i]
        p_D=lognormal(D, mu, sigma)
        loglike+=np.log(p_D)
    return loglike


def planck_lowT_binned_loglike_cobaya(_theory={'Cl': {'tt': 29}}):
    ell = _theory.get_Cl(ell_factor=False)['ell'][2:30]
    Cl_theory = _theory.get_Cl(ell_factor=False)['tt'][2:30]
    Dl_theory_bin=get_binned_D_from_theory_Cls(ell , Cl_theory, lmin_list_TT, lmax_list_TT)
    loglike=0
    for i, D in enumerate(Dl_theory_bin):
        mu=mu_LN_TT[i]
        sigma=sig_LN_TT[i]
        p_D=lognormal(D, mu, sigma)
        loglike+=np.log(p_D)
    return loglike
