import glob
import os
import numpy as np
from matplotlib.pylab import *
from matplotlib.image import NonUniformImage
from scipy.optimize import leastsq

# Reexport for convenience
from GYSutils_hdf5 import *

# Test Python version for 'input' definition
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
_use_default = False
if PY2:
    import __builtin__
    __builtin__.input = __builtin__.raw_input

    if os.environ.has_key("GYS_POSTPROC_USE_DEFAULTS") and \
       os.environ.get("GYS_POSTPROC_USE_DEFAULTS") == '1':
        _use_default = True

if PY3:
    if os.environ.get("GYS_POSTPROC_USE_DEFAULTS") == '1':
        _use_default = True


def input_default(msg):
    if "default" in msg:
        print(msg+' *default*')
        return ''
    else:
        return input(msg)

if _use_default:
    gys_input = input_default
else:
    gys_input = input

#---------------------------------------
# Colors class
#---------------------------------------
class bcolors:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    GREEN   = '\033[92m'
    WARNING = '\033[96m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'

    def disable(self):
        self.HEADER  = ''
        self.BLUE    = ''
        self.GREEN   = ''
        self.WARNING = ''
        self.FAIL    = ''
        self.ENDC    = ''
#end class bcolors


#--------------------------------------------------
# Definition of the dictionnary containing the
# equivalence between old and new names of output variables
#--------------------------------------------------
def define_corresp_oldnew_names():

    """
    Definition of the dictionnary containing the
    equivalence between old and new names of output variables
    """

    dic_corresp_oldnew_names = {}
    dic_corresp_oldnew_names = {
        #--> Modification of the names of the guiding-center quantities
        #-- OLD NAMES         :  NEW NAMES
        'niGCr'               :'densGC_FSavg',
        'nvpoloGCr_turb'      :'nvpoloGC_turb_FSavg',
        'nvpoloGCr_neo'       :'nvpoloGC_neo_FSavg',
        'nvpoloGCr_vpar'      :'nvpoloGC_vpar_FSavg',
        'pressGCr_perp'       :'pressGC_perp_FSavg',
        'pressGCr_par'        :'stressGC_par_FSavg',
        'LphiGCr'             :'LtorGC_FSavg',
        'TpolGCr'             :'TpolGC_psi_FSavg',
        'TpolGC_chi_FSavg'    :'TpolGC_psi_FSavg',
        'GammaGCr_turb'       :'GammaGC_turb_r_FSavg',
        'GammaGCr_neo'        :'GammaGC_neo_r_FSavg',
        'RSphiGCr_turb'       :'RSphiGC_turb_r_FSavg',
        'RSphiGCr_neo'        :'RSphiGC_neo_r_FSavg',
        'dLphiGCr_dt'         :'dLtorGC_dt_FSavg',
        'QGCr_perp_turb'      :'QGC_perp_turb_r_FSavg',
        'QGCr_perp_neo'       :'QGC_perp_neo_r_FSavg',
        'QGCr_par_turb'       :'QGC_par_turb_r_FSavg',
        'QGCr_par_neo'        :'QGC_par_neo_r_FSavg',
        'QGCr_dtvpar'         :'QGC_dtvpar_FSavg',
        'dpGCr_dt_perp'       :'dpressGC_dt_perp_FSavg',
        'dpGCr_dt_par'        :'dstressGC_dt_par_FSavg',
        'SGCr_pperp_par'      :'SGC_Pperp_par_FSavg',
        'SGCr_pperp_turb'     :'SGC_Pperp_turb_FSavg',
        'heatexGCr_turb'      :'heatexGC_turb_FSavg',
        'heatexGCr_neo'       :'heatexGC_neo_FSavg',
        'heatexGCr_turb_diag' :'heatexGC_turb_FSavg',
        'heatexGCr_neo_diag'  :'heatexGC_neo_FSavg',
        'QGC_loc'             :'QGC_rtheta',
        'RSphiGC_loc'         :'RSphiGC_rtheta',
        'entropy'             :'entropy_tot',
        'L2norm'              :'L2norm_tot',
        'Enkin'               :'deltaEnkin_tot',
        #--> Modification of the names of the particle quantities
        'nir'                 :'dens_FSavg',
        'nuparr'              :'nupar_FSavg',
        'nvpolor_mag'         :'nvpolo_mag_FSavg',
        'pressr'              :'stress_FSavg',
        'Qr_turb'             :'Q_turb_r_FSavg',
        'Qr_neo'              :'Q_neo_r_FSavg',
        'Gammar_turb'         :'Gamma_turb_r_FSavg',
        'Gammar_neo'          :'Gamma_neo_r_FSavg',
        'RSthetar'            :'RStheta_r_FSavg',
        'RSparr'              :'RSpar_r_FSavg',
        'Enpot'               :'deltaEnpot_tot',
        #--> Modification of the names for the collision operators
        'Tmean_r'             :'Tcoll_r',
        'Vmean_r'             :'Vcoll_r',
        'Tmean_r_avg'         :'Tcoll_r_avg',
        'Vmean_r_avg'         :'Vcoll_r_avg',
        'Tmean_thetaphi'      :'Tcoll_thetaphi',
        'Vmean_thetaphi'      :'Vcoll_thetaphi',
        #--> Modification of the names for distribution function
        #    cross-sections
        'frthvm'              :'frtheta_passing',
        'frvpar_mu0'          :'frvpar_passing',
        'fphivpar_mu0'        :'fphivpar_passing',
        'fthvpar_mu0'         :'fthvpar_passing',
        'frthv0'              :'frtheta_trapped',
        'frvpar_mumax'        :'frvpar_trapped',
        'fphivpar_mumax'      :'fphivpar_trapped',
        'fthvpar_mumax'       :'fthvpar_trapped',
        #--> Exceptional modifications
        'nbions'              :'nbion',
        'enveloppe_r'         :'envelope_r',
        #--> For geometry
        'small_radius'        :'minor_radius',
        }
    return dic_corresp_oldnew_names

#end def define_oldnew_names


#--------------------------------------------------
# Definition of the dictionnary containing the
#  equivalence between old and new names for
#  neoclassical and turbulent output variables
#--------------------------------------------------
def define_corresp_oldnew_neoturb_names():

    """
    Definition of the dictionnary containing the
     equivalence between old and new names for
     neoclassical and turbulent output variables
    """

    dic_corresp_oldnew_neoturb_names = {}
    dic_corresp_oldnew_neoturb_names = {
        #--> Modification of the names for the guiding-centers
        #-- OLD NAMES         :  NEW NAMES
        'nvpoloGC_turb_FSavg'   : 'nvpolGC_vE_FSavg',
        'nvpoloGC_neo_FSavg'    : 'nvpolGC_vD_FSavg',
        'GammaGC_turb_r_FSavg'  : 'GammaGC_vE_r_FSavg',
        'GammaGC_neo_r_FSavg'   : 'GammaGC_vD_r_FSavg',
        'RSphiGC_turb_r_FSavg'  : 'RSphiGC_vE_r_FSavg',
        'RSphiGC_neo_r_FSavg'   : 'RSphiGC_vD_r_FSavg',
        'QGC_perp_turb_r_FSavg' : 'QGC_perp_vE_r_FSavg',
        'QGC_perp_neo_r_FSavg'  : 'QGC_perp_vD_r_FSavg',
        'QGC_par_turb_r_FSavg'  : 'QGC_par_vE_r_FSavg',
        'QGC_par_neo_r_FSavg'   : 'QGC_par_vD_r_FSavg',
        'QGC_dtvpar_FSavg'      : 'QGC_dtvpar_FSavg',
        'SGC_Pperp_turb_FSavg'  : 'SGC_Pperp_vE_FSavg',
        'heatexGC_turb_FSavg'   : 'heatexGC_vE_FSavg',
        'heatexGC_neo_FSavg'    : 'heatexGC_vD_FSavg',
        'heatexGC_turb_FSavg'   : 'heatexGC_vE_FSavg',
        'heatexGC_neo_FSavg'    : 'heatexGC_vD_FSavg',
        'QGC_rtheta'            : 'QGC_vE_rtheta',
        'RSphiGC_rtheta'        : 'RSphiGC_vE_rtheta',
        #--> Modification of the names of the particle quantities
        'Q_turb_r_FSavg'        : 'Q_vE_r_FSavg',
        'Q_neo_r_FSavg'         : 'Q_vD_r_FSavg',
        'Gamma_turb_r_FSavg'    : 'Gamma_vE_r_FSavg',
        'Gamma_neo_r_FSavg'     : 'Gamma_vD_r_FSavg',
        'RStheta_r_FSavg'       : 'RStheta_vE_r_FSavg',
        #--> Modification of nvpolo into nvpol
        'nvpoloGC_vpar_FSavg'   : 'nvpolGC_vpar_FSavg',
        'nvpolo_mag_FSavg'      : 'nvpol_mag_FSavg',
        #--> Other modifications
        'nupar_FSavg'           : 'nVpar_FSvg'
        }

    return dic_corresp_oldnew_neoturb_names

#end def define_oldnew_neoturb_names
#--------------------------------------------------


#--------------------------------------------------
# import or reload a module already existent
#  Input:
#    - modulename (str) : name of the module
#--------------------------------------------------
def impall(modulename):

    """ Import or reload a module already existent"""
    import sys
    from imp import reload

    exist_module = modulename in sys.modules.keys()

    exec("import {mod}".format(mod=modulename), globals(), locals())
    if (exist_module):
        exec("reload({mod})".format(mod=modulename), globals(), locals())

    exec("global module_load; module_load = {mod}".format(mod=modulename), globals(), locals())

    return module_load

#end def impall
#----------------------------------------------


#-------------------------------------------------
#--> Find negative values of 'varname' variable
#     in 'filename' file
#-------------------------------------------------
def Find_NegativValues(filename, varname):

    """
    Find negative values of 'varname' variable
    in 'filename' file
    """

    fh5 = loadHDF5(filename)
    H5f_var = getattr(fh5, varname)
    findneg_val = (H5f_var < 0.).nonzero()
    nbneg_val   = np.shape(findneg_val)[1]
    return [nbneg_val, findneg_val]

#end def Find_NegativValues
#----------------------------------------------

#---------------------------------------------
# Ask the initial and the final times
#---------------------------------------------
def Ask_firstandlast_time(Ttime, ask_diagstep=True, msg=''):

    """ Ask the initial and the final times"""

    nb_time = np.size(Ttime)
    tinit   = gys_input(' init time ' + msg +
                        ' (between ' + str(Ttime[0]) +
                        ' and ' + str(Ttime[nb_time - 1]) +
                        ' (default ' + str(Ttime[0]) + ')) ? : ')
    if (tinit != ''):
        t_init = str2num(tinit)
    else:
        t_init = Ttime[0]
    it1 = search_pos(t_init, Ttime)

    tfin = gys_input(' end time ' + msg +
                     ' (between ' + str(Ttime[it1]) +
                     ' and ' + str(Ttime[nb_time - 1]) +
                     '  (default ' + str(Ttime[nb_time - 1]) + ')) ? : ')
    if (tfin != ''):
        t_fin = str2num(tfin)
    else:
        t_fin = Ttime[nb_time - 1]
    it2 = search_pos(t_fin, Ttime)

    if (ask_diagstep):
        if (it1 != it2):
            time_step = gys_input(' --> Diag step (between 1 and ' +
                                  str(nb_time / 3) + ') ? (default 1) : ')
            if (time_step != ''):
                istep = str2num(time_step)
            else:
                istep = 1
        else:
            istep = 1
        return [it1, it2, istep]
    else:
        return [it1, it2]

#end def Ask_firstandlast_time
#----------------------------------------------


#-----------------------------------------------------------------
# Ask for a specific time
#  -> Default value is given in function of 'give_default' value,
#      . give_default = 0 : no default value
#      . give_default = 1 : first time by default
#      . give_default = 2 : last time by default
#-----------------------------------------------------------------
def Ask_time(Ttime, ask_msg='', first_indx_time=0, give_default=0):

    """
    Ask for a specific time
    -> Default value is given in function of 'give_default' value,
    . give_default = 0 : no default value
    . give_default = 1 : first time by default
    . give_default = 2 : last time by default
    """

    nb_time = np.size(Ttime)
    if (ask_msg == ''):
        ask_msg = ' Time'

    if (give_default == 0):
        str_time_ask = gys_input(ask_msg +
                                 ' (between ' + str(Ttime[first_indx_time]) +
                                 ' and ' + str(Ttime[nb_time - 1]) + ') ? : ')
        if (str_time_ask != ''):
            time_ask      = str2num(str_time_ask)
            int_time_find = search_pos(time_ask, Ttime)
        else:
            int_time_find = nb_time + 10
    #end if

    if (give_default > 0):
        if (give_default == 1):
            indx_default = first_indx_time
        if (give_default == 2):
            indx_default = nb_time - 1
        str_time_ask = gys_input(ask_msg +
                                 ' (between ' + str(Ttime[first_indx_time]) +
                                 ' and ' + str(Ttime[nb_time - 1]) +
                                 ' [default ' + str(Ttime[indx_default]) +
                                 ']) ? : ')
        if (str_time_ask != ''):
            time_ask      = str2num(str_time_ask)
            int_time_find = search_pos(time_ask, Ttime)
        else:
            if (give_default == 1):
                time_ask = Ttime[first_indx_time]

            if (give_default == 2):
                time_ask = Ttime[nb_time - 1]

            int_time_find = search_pos(time_ask, Ttime)
        #end if
    #end if

    if ( int_time_find<=nb_time ):
        str_time_find = str(Ttime[int_time_find])
    else:
        str_time_find = ""
    #end if

    return [str_time_find, int_time_find]

#end def Ask_time
#----------------------------------------------

#-----------------------------------------------------------------
# Ask for a specific normalized radial position
#  -> Default value is given in function of 'give_default' value,
#      . give_default = 0 : no default value
#      . give_default = 1 : medium value
#  (developed by Y. Sarazin)
#-----------------------------------------------------------------
def Ask_rho(Rrho, ask_msg='', first_indx_rho=0, give_default=0):

    """
    Ask for a specific normalized radius (rho)
    -> Default value is given in function of 'give_default' value,
    . give_default = 0 : no default value
    . give_default = 1 : medium rho by default
    """

    nb_rho = np.size(Rrho)
    if (ask_msg == ''):
        ask_msg = ' Rho'

    if (give_default == 0):
        str_rho_ask = gys_input(ask_msg +
                                 ' (between ' + str(Rrho[first_indx_rho]) +
                                 ' and ' + str(Rrho[nb_rho - 1]) + ') ? : ')
        if (str_rho_ask != ''):
            rho_ask      = str2num(str_rho_ask)
            int_rho_find = search_pos(rho_ask, Rrho)
        else:
            int_rho_find = nb_rho + 10
    #end if

    if (give_default > 0):
        if (give_default == 1):
            indx_default = nb_rho//2
        str_rho_ask = gys_input(ask_msg +
                                 ' (between ' + str(Rrho[first_indx_rho]) +
                                 ' and ' + str(Rrho[nb_rho - 1]) +
                                 ' [default ' + str(Rrho[indx_default]) +
                                 ']) ? : ')
        if (str_rho_ask != ''):
            rho_ask      = str2num(str_rho_ask)
            int_rho_find = search_pos(rho_ask, Rrho)
        else:
            if (give_default == 1):
                rho_ask = Rrho[indx_default]

            int_rho_find = search_pos(rho_ask, Rrho)
        #end if
    #end if

    if ( int_rho_find<=nb_rho ):
        str_rho_find = str(Rrho[int_rho_find])
    else:
        str_rho_find = ""
    #end if

    return [str_rho_find, int_rho_find]

#end def Ask_rho
#----------------------------------------------


#----------------------------------------------
# Find first and last diagnostics which are
#  presents in the directory
#----------------------------------------------
def Find_FirstAndLastDiag(ResuFilenames, ExtraChar='d', ExceptFileName=None):
    """ Find first and last diagnostics which
    are presents in the directory"""

    #---> List of file names
    resu_fnames = glob.glob(ResuFilenames)
    if (ExceptFileName != None):
        if (ExceptFileName in resu_fnames):
            resu_fnames.remove(ExceptFileName);
    resu_fnames.sort()

    #---> Find the first and last files present
    str_tmp        = os.path.basename(resu_fnames[0])
    str_tmp        = str_tmp.split('_')[-1]
    str_first_file = str_tmp.split('.h5')[0].split(ExtraChar)[-1]
    str_tmp        = os.path.basename(resu_fnames[-1])
    str_tmp        = str_tmp.split('_')[-1]
    str_last_file  = str_tmp.split('.h5')[0].split(ExtraChar)[-1]

    return [str_first_file,str_last_file]

#end def Find_FirstAndLastDiag
#----------------------------------------------


#--------------------------------------------------
# tw is equal to :
#  True if the variables are keeped in the
#  dictionarie (default)
#  False if the variables are defined as global
#--------------------------------------------------
def DctCorr(dct, tw=True):
    """
    Check if variable is in the dictionarie or global
    tw is equal to :
    True if the variables are keeped in the dictionarie (default)
    False if the variables are defined as global
    """

    dctret = {}
    for k in list(dct.keys()):
        idx = k.find('\x00')
        if idx != -1:
            k2 = k[:idx]
        else:
            k2 = k
        dctret[k2] = dct[k]
    if tw:
        glo = globals()
        for k in list(dctret.keys()):
            glo[k] = dctret[k]
    return dctret

#end def DctCorr
#--------------------------------------------------

#--------------------------------------------------
# definition of the size of the axes
#--------------------------------------------------
def my_axes(ax, fs=16):

    """ Definition of the size of the axes"""

    xticklabels = get(ax, 'xticklabels')
    yticklabels = get(ax, 'yticklabels')
    setp(xticklabels, 'color', 'k', fontsize=fs)
    setp(yticklabels, 'color', 'k', fontsize=fs)

#end def my_axes
#--------------------------------------------------


#--------------------------------------------------
# definition of the size of the legend
#--------------------------------------------------
def my_legend(fs=16, visibl=True, lo='best'):

    """ Definition of the size of the legend"""

    leg   = legend(loc=lo)
    ltext = leg.get_texts()
    setp(ltext, 'fontsize', fs)
    lframe = leg.get_frame()
    setp(lframe, fc='w', ec='w', visible=visibl)
    return leg

#end def my_legend
#--------------------------------------------------


#--------------------------------------------------
# definition of the size of xlabel
#--------------------------------------------------
def my_xlabel(lab="", s=18, tex=0):

    """ Definition of the size of xlabel"""

    if (tex == 0):
        xlab = xlabel(r'$' + lab + '$', size=s)
    else:
        xlab = xlabel(lab, size=s)
    return xlab

#end def my_xlabel
#--------------------------------------------------


#--------------------------------------------------
# definition of the size of ylabel
#--------------------------------------------------
def my_ylabel(lab="", s=18, tex=0):

    """ Definition of the size of ylabel"""

    if (tex == 0):
        ylab = ylabel(r'$' + lab + '$', size=s)
    else:
        ylab = ylabel(lab, size=s)
    return ylab

#end def my_ylabel
#--------------------------------------------------


#--------------------------------------------------
# definition of the size of title
#--------------------------------------------------
def my_title(lab="", s=16, tex=0):

    """ Definition of the size of title"""

    if (tex == 0):
        tit = title(r'$' + lab + '$', size=s)
    else:
        tit = title(lab, size=s)
    return tit

#end def my_title
#--------------------------------------------------


#--------------------------------------------------
# Title time
#--------------------------------------------------
def my_suptitle(lab="", s=18, tex=0):

    """ Title time"""

    if (tex == 0):
        tit = suptitle(r'$' + lab + '$', size=s)
    else:
        tit = suptitle(lab, size=s)
    return tit

#end def title_time
#--------------------------------------------------


#--------------------------------------------------
# definition of the Gray map
#--------------------------------------------------
def my_graymap():

    """ Definition of the Gray map"""

    graymap = get_cmap('gray')
    graymap._segmentdata = {'blue': ((0.0, 1, 1), (1.0, 0, 0)),
                            'green': ((0.0, 1, 1), (1.0, 0, 0)),
                            'red': ((0.0, 1, 1), (1.0, 0, 0))}
    return graymap

#end def my_graymap
#--------------------------------------------------


#--------------------------------------------------
# definition of the format of the colorbar
#--------------------------------------------------
def my_colorbar():

    """ Definition of the format of the colorbar"""

    # 2.041
    #colorbar(format='%.3f')
    # 2.0e+00
    colorbar(format='%1.1e')
    # 2.0e+00
    #colorbar(format='%24.12e')

#end def my_colorbar
#--------------------------------------------------


#--------------------------------------------------
# definition of the close('all')
#--------------------------------------------------
def closa():

    """ Definition of the close('all')"""

    close('all')

#end def closa
#--------------------------------------------------


#--------------------------------------------------
# search index position
#--------------------------------------------------
def search_pos(x_value, x_array):
    """ Search index position"""

    x_indx = int(np.searchsorted(x_array, x_value, side='left'))

    return x_indx
#end def search_pos
#--------------------------------------------------


#-------------------------------------------------
# First Derivative
#   Input: F        = function to be derivate
#          dx       = step of the variable 
#                      for derivative
#          periodic = 1 if F is periodic
#                   = 0 otherwise (by default)
#   Output: dFdx = first derivative of F
#-------------------------------------------------
def Derivee1(F, dx, periodic=False, axis=0):

    """
    First Derivative
       Input: F        = function to be derivate
              dx       = step of the variable for derivative
              periodic = 1 if F is periodic
       Output: dFdx = first derivative of F
    """
    if axis != 0:
        F = np.swapaxes(F, axis, 0)
        F = Derivee1(F, dx, periodic=periodic, axis=0)
        F = np.swapaxes(F, axis, 0)
        return F

    nx   = np.size(F, 0)
    dFdx = np.empty_like(F)

    c0 = 2. / 3.
    dFdx[2:-2] = c0 / dx * (
        F[3:-1] - F[1:-3] - (F[4:] - F[:-4]) / 8
    )

    c1 = 4. / 3.
    c2 = 25. / 12.
    c3 = 5. / 6.
    if not periodic:
        dFdx[0]  = (
            - F[4] / 4.
            + F[3] * c1
            - F[2] * 3.
            + F[1] * 4.
            - F[0] * c2
        ) / dx
        dFdx[-1] = (
            + F[-5] / 4.
            - F[-4] * c1
            + F[-3] * 3.
            - F[-2] * 4.
            + F[-1] * c2
        ) / dx
        dFdx[1]  = (
            + F[4] / 12.
            - F[3] / 2.
            + F[2] / c0
            - F[1] * c3
            - F[0] / 4.
        ) / dx
        dFdx[-2] = (
            - F[-5] / 12.
            + F[-4] / 2.
            - F[-3] / c0
            + F[-2] * c3
            + F[-1] / 4.
        ) / dx
    else:
        # Here, we need to take care of the ghost point on the edge
        dFdx[ 1] = c0 / dx * (F[2] - F[-1] - (F[3] - F[-2]) / 8.)
        dFdx[ 0] = c0 / dx * (F[1] - F[-2] - (F[2] - F[-3]) / 8)
        dFdx[-1] = dFdx[0]
        dFdx[-2] = c0 / dx * (F[0] - F[-3] - (F[1] - F[-4]) / 8.)

    return dFdx

#end def Derivee1
#--------------------------------------------------


#--------------------------------------------------
#  First derivative for a 2D array
#--------------------------------------------------
def Derivee1_2d(F, dt, periodic=0):
    """
    First Derivative
       Input: F(nt,nx)        = function to be derivate
              dt(nt)          = step of the variable for derivative
              periodic = 1 if F is periodic
       Output: dFdt = first derivative of F along t (axis=0)
    """
    return Derivee1(F, dt, periodic, axis=0)

#end def Derivee1_2d
#--------------------------------------------------


#--------------------------------------------------
# Second Derivative
#   Input: F        = function to be derivate
#          dx       = step of the variable 
#                      for derivative
#          periodic = 1 if F is periodic
#                   = 0 otherwise (by default)
#   Output: dFdx = second derivative of F
#------------------------------------------------------------
def Derivee2(F, dx, periodic=0, axis=0):

    """
    Second Derivative
      Input: F        = function to be derivate
             dx       = step of the variable for derivative
             periodic = 1 if F is periodic
                      = 0 otherwise (by default)
      Output: dFdx = second derivative of F
    """
    if axis:
        F = np.swapaxes(F, 0, axis)
        F = Derivee2(F, dx, periodic, axis=0)
        F = np.swapaxes(F, 0, axis)
        return F

    dx2 = dx * dx

    d2Fdx = np.empty_like(F)
    d2Fdx[2:-2] = (
        - 30. *  F[2:-2]
        + 16. * (F[3:-1] + F[1:-3])
        -       (F[0:-4] + F[4:  ])
    ) / (12. * dx2)

    c1 = 11. / 12.
    c2 = 14. / 3.
    c3 = 9.5
    c4 = 26. / 3.
    c5 = 35. / 12.
    c6 = 5. / 3.
    c7 = 11. / 12.
    if (periodic == 0):
        d2Fdx[ 0] = (c1 * F[ 4] - c2 * F[ 3] + c3 * F[ 2] - c4 * F[ 1] + c5 * F[ 0]) / dx2
        d2Fdx[-1] = (c1 * F[-5] - c2 * F[-4] + c3 * F[-3] - c4 * F[-2] + c5 * F[-1]) / dx2
        d2Fdx[ 1] = (-F[ 4] / 12. + F[ 3] / 3. + F[ 2] / 2. - c6 * F[ 1] + c7 * F[ 0]) / dx2
        d2Fdx[-2] = (-F[-5] / 12. + F[-4] / 3. + F[-3] / 2. - c6 * F[-2] + c7 * F[-1]) / dx2
    else:
        d2Fdx[1]  = (-30. * F[ 1] + 16. * (F[ 2] + F[-1]) - (F[-2] + F[3])) / (12. * dx2)
        d2Fdx[0]  = (-30. * F[ 0] + 16. * (F[ 1] + F[-2]) - (F[-3] + F[2])) / (12. * dx2)
        d2Fdx[-2] = (-30. * F[-2] + 16. * (F[-1] + F[-3]) - (F[-4] + F[1])) / (12. * dx2)
        d2Fdx[-1] = d2Fdx[0]
    #end if

    return d2Fdx

#end def Derivee2
#--------------------------------------------------

#-------------------------------------------------
# Personal FFT1D function
#-------------------------------------------------
def Fourier1D(F0, x0, axis=-1):
    """ Personal FFT1D function"""
    x0 = np.asarray(x0)
    assert x0.ndim == 1

    nx0 = len(x0)
    nx  = 2 * int(nx0 / 2)
    hnx = int(nx / 2)

    dx = x0[1] - x0[0]
    kx = 2 * np.pi / dx * np.linspace(-.5, .5, nx, endpoint=False)

    TFF  = np.conjugate(np.fft.fft(
        np.conjugate(F0),
        n=nx, # crop `axis` at length `nx`
        axis=axis
    ))
    TFF /= nx
    TFF  = np.fft.fftshift(TFF, axes=axis)

    return TFF, kx

#end def Fourier1D
#--------------------------------------------------


#-------------------------------------------------
# Personal FFT2D function
#-------------------------------------------------
def Fourier2D(F0, y0, x0):

    """ Personal FFT2D function"""

    nx0 = len(x0)
    nx  = 2 * int(nx0 / 2)
    hnx = int(nx / 2)
    ny0 = len(y0)
    ny  = 2 * int(ny0 / 2)
    hny = int(ny / 2)

    x = x0[0:nx]
    y = y0[0:ny]
    F = F0[0:ny, 0:nx]

    Lx   = x[nx - 1] - x[0]
    dx   = x[1] - x[0]
    dkx  = 2. * pi / (Lx + dx)
    kx   = np.zeros(nx)
    temp = -dkx * r_[1:hnx + 1]
    kx[0:hnx]  = temp[::-1]
    kx[hnx:nx] = dkx * r_[0:hnx]

    Ly   = y[ny - 1] - y[0]
    dy   = y[1] - y[0]
    dky  = 2. * pi / (Ly + dy)
    ky   = np.zeros(ny)
    temp = -dky * r_[1:hny + 1]
    ky[0:hny]  = temp[::-1]
    ky[hny:ny] = dky * r_[0:hny]

    TFF = np.zeros((ny, nx), dtype=complex)
    AA  = np.zeros((ny, nx), dtype=complex)
    var = np.conjugate(np.fft.fft2(np.conjugate(F))) / float((nx * ny))

    AA[:, 0:hnx]   = var[:, hnx:nx]
    AA[:, hnx:nx]  = var[:, 0:hnx]
    TFF[0:hny, :]  = AA[hny:ny, :]
    TFF[hny:ny, :] = AA[0:hny, :]

    return TFF, kx, ky
#end def Fourier2D
#--------------------------------------------------

#-------------------------------------------------
# Personal inverse FFT1D function
#-------------------------------------------------
def iFourier1D(TFF, kx, axis=-1, pad=False, real=True):
    """ Personal inverse FFT1D function"""
    kx = np.asarray(kx)
    assert kx.ndim == 1

    nx  = len(kx)
    hnx = int(nx / 2)
    dkx = abs(kx[1] - kx[0])
    x   = np.linspace(0, 2 * np.pi / dkx, nx, endpoint=False)

    TFF = np.fft.ifftshift(TFF, axes=axis)
    F = np.fft.ifft(nx * np.conjugate(TFF), axis=axis)
    if real:
        F = F.real
    else:
        F = F.conj()
    #end if

    # Add the redundant point on the transformation axis
    if pad:
        pad_width = np.zeros((F.ndim, 2)).astype('int')
        pad_width[axis, 1] = 1
        F = np.pad(F, pad_width, 'wrap')
    #end if

    return F, x
#end def iFourier1D
#--------------------------------------------------

#-------------------------------------------------
# Personal inverse FFT2D function
#-------------------------------------------------
def iFourier2D(TFF, ky, kx, axes=(0, 1), pad=(False, False), real=True):
    """ Personal FFT2D inverse function"""
    kx = np.asarray(kx)
    assert kx.ndim == 1
    ky = np.asarray(ky)
    assert ky.ndim == 1

    nx  = len(kx)
    hnx = int(nx / 2)
    dkx = abs(kx[1] - kx[0])
    x   = r_[0:nx] * 2. * pi / (nx * dkx)

    ny  = len(ky)
    hny = int(ny / 2)
    dky = abs(ky[1] - ky[0])
    y   = np.linspace(0, 2 * np.pi / dky, ny, endpoint=False)

    TFF = np.fft.ifftshift(TFF, axes=axes)
    F = np.fft.ifft2(nx * ny * np.conjugate(TFF), axes=axes)
    if real:
        F = F.real
    else:
        F = F.conj()
    #end if

    # Add the redundant point on the transformation axis
    if any(pad):
        pad_width = np.zeros(F.ndim, 2)
        pad_width[axes, 1] = pad
        F = np.pad(F, pad_width, 'wrap')
    #end if

    return F, x, y
#end def iFourier2D
#--------------------------------------------------


#--------------------------------------------------
# Compute moving average of a signal
#--------------------------------------------------
def Moving_average(F,lamb, axis=0):
    sh = F.shape
    mean_sh = np.zeros(sh[axis])
    for n in range(len(sh[axis])):
        mean_sh[n] = np.mean(F[n:lamb+n],axis=axis)
    return mean_sh

#end def Moving_average
#--------------------------------------------------


#--------------------------------------------------
# Compute rms
#--------------------------------------------------
def rms(W):

    rmsW = np.sqrt(np.mean((W-np.mean(W))*(W-np.mean(W))))

    return rmsW
#--------------------------------------------------


#--------------------------------------------------
# Least square linear regression
#--------------------------------------------------
def Least_square_LR(xdata,ydata):

    fitfunction = lambda p, x : p[0]*x+p[1]
    errfunc = lambda p, x, y: fitfunction(p,x) - y
    p0 = [-0.2,8]
    p1,success = leastsq(errfunc,p0[:],args=(xdata,ydata))

    return p1

#def Least_square_LR
#--------------------------------------------------


#--------------------------------------------------
# 1D Probability Distribution Function
#--------------------------------------------------
def PDFfunction(W):

    max_W = np.max(W)
    min_W = np.min(W)
    rms_W = rms(W)
    bins = np.linspace(min_W,max_W,rms_W/5)
    NDF = np.zeros(len(bins))
    for i in W:
        idx = np.where(np.abs(bins-i)-min(np.abs(bins-i))==0)[0]
        NDF[idx] = NDF[idx] + 1

    return NDF,bins

#end def PDFfunction
#--------------------------------------------------


#--------------------------------------------------
# 2D Probability Distribution function
#--------------------------------------------------
def PDFfunction2D(W):

    [nt,nr] = W.shape
    ir = int(2*nr/3)
    max_W = np.max(W)/20
    min_W = 0
    rms_W = rms(W[:,ir])
    mean_W = np.mean(W)
    step = rms_W/20
    max_bins = mean_W + 100*step
    min_bins = mean_W - 100*step
    BINS = np.arange(0,max_bins,step)
    bins = np.ones((nr,len(BINS)))*BINS
    NDF = np.zeros(bins.shape)
    for ii in range(nt):
        y = np.argmin(np.abs(bins.T-W[ii,:]),axis=0)
        NDF[np.arange(NDF.shape[0]), y] += 1

    return NDF, BINS

#end def PDFfunction2D
#--------------------------------------------------


#--------------------------------------------------
#  Find the minimum and maximum indices in a array
#   corresponding to the minimal and maximal values
#   asked
#--------------------------------------------------
def find_min_max(x_array, x_name):

    """
    Find the minimum and maximum indices in a array
    corresponding to the minimal and maximal values
    asked
    """

    xinf_str = x_array[0]
    xsup_str = x_array[size(x_array) - 1]
    xmin_str = gys_input('minimum ' + x_name +
                         ' (' + repr(xinf_str) + ' to ' +
                         repr(xsup_str) + ') [default ' +
                         repr(xinf_str) + '] ? : ')
    xmax_str = gys_input('maximum ' + x_name +
                         ' (' + repr(xinf_str) + ' to ' + repr(xsup_str) +
                         ') [default ' + repr(xsup_str) + '] ? : ')

    if (xmin_str == ''):
        indx_min = 0
    else:
        xmin     = float(xmin_str)
        indx_min = find(x_array <= xmin)
        indx_min = indx_min[size(indx_min) - 1]

    if (xmax_str == ''):
        indx_max = size(x_array) - 1
    else:
        xmax     = float(xmax_str)
        indx_max = find(x_array >= xmax)
        indx_max = indx_max[0]

        return indx_min, indx_max

#end def find_min_max
#--------------------------------------------------


#--------------------------------------------------
#   find the path for executable file
#--------------------------------------------------
def my_execfile(filename):

    """ Find the path for executable file"""

    import os
    my_path     = os.environ.get('PYTHONPATH')
    directories = my_path.split(os.pathsep)
    ifind = 0
    for idir in directories:
        filesearch = idir + '/' + filename
        if os.path.isfile(filesearch):
            ifind = 1
            exec(compile(open(filesearch).read(), filesearch, 'exec'), globals(), locals())
    if ifind == 0:
        print(bcolors.FAIL+' filename not found !! '+bcolors.ENDC)

#end def my_execfile
#--------------------------------------------------


#--------------------------------------------------
# Print the dictionnary
#--------------------------------------------------
def printDict(di, format=" %-25s %s"):

    """ Print the dictionnary"""

    for (key, val) in list(di.items()):
        print(format % (str(key) + ':', val))

#end def printDict
#--------------------------------------------------


#--------------------------------------------------
#  Construct the file name according to the number
#    ifile
#--------------------------------------------------
def create_filename(acronym, ifile):
    """ Construct the file name according
    to the number ifile"""

    file_name = '{acronym}_{ifile:04d}.mat'.format(
        acronym=acronym,ifile=ifile)
    return file_name
#end def create_filename
#--------------------------------------------------


#--------------------------------------------------
#  Construct the file name according to the number
#    ifile
#--------------------------------------------------
def create_diagname(acronym, ifile):
    """ Construct the file name according
    to the number ifile"""

    file_name = '{acronym}_{letter}{ifile:05d}.h5'.format(
        acronym=acronym,letter='d',ifile=ifile)
    return file_name

#end def create_diagname
#--------------------------------------------------


#--------------------------------------------------
# Test for non-uniform images
#--------------------------------------------------
def nonuniform_imshow(x, y, C, **kwargs):

    """
    Plot image with nonuniform pixel spacing.
    This function is a convenience method for
    calling image.NonUniformImage.
    """
    ax = plt.gca()
    im = NonUniformImage(ax, **kwargs)
    im.set_data(x, y, C)
    ax.images.append(im)
    return im

#end def nonuniform_imshow
#--------------------------------------------------


#--------------------------------------------------
# To convert string to integer or float
#--------------------------------------------------
def str2num(datum):

    """ Convert string to integer or float"""

    try:
        return int(datum)
    except:
        try:
            return float(datum)
        except:
            return datum

#end def str2num
#--------------------------------------------------


#--------------------------------------------------
# Write in a file
#--------------------------------------------------
def write_log(file_log, s):

    """ Write in a file"""

    F = open(file_log, 'a')

    if (not os.path.exists(file_log)):
        print(str(s))
    else:
        F.write(s + str("\n"))

    F.close()

#end def write_log
#--------------------------------------------------


#--------------------------------------------------
#
#--------------------------------------------------
def poloidal_plot(r, t, y, colmap='bwr', separatrix=0):
    """ Poloidal plot"""

    amplitude = np.amax(np.abs(y))
    xx = dot(cos(t.reshape(len(t), 1)), r.reshape(1, len(r)))
    yy = dot(sin(t.reshape(len(t), 1)), r.reshape(1, len(r)))
    y_remap = copy(y.reshape((len(t), len(r))))

    if (colmap=='bwr'):
        p = pcolormesh(xx, yy, y_remap, vmin = -amplitude, vmax = amplitude, cmap=colmap)
    else:
        p = pcolormesh(xx, yy, y_remap, vmin = 0, vmax = amplitude, cmap=colmap)

    if (separatrix==1):
        sep = plot(sp0.LIM_rseparatrix*sp0.a*np.sin(sp0.thetag),
                   sp0.LIM_rseparatrix*sp0.a*np.cos(sp0.thetag),
                   color='k',ls='--',lw=1.5)
    axis('equal')
    ax = gca()
    ax.set_xticks([])
    ax.set_yticks([])
    axis('tight')
    colorbar()

#end def poloidal_plot
#-----------------------------------------------------


#-----------------------------------------------------
# Figure saving
#-----------------------------------------------------
def save_figure(fname,directory):

    savefig(directory+fname+'.png', dpi=200)
    print(fname+'.png is saved in '+directory)

#end def save_figure
#-----------------------------------------------------


#-----------------------------------------------------
# Define graphic axis
#-----------------------------------------------------
def mefAxes( ax, fs=16, clX=(0,0,0,1), clY=(0,0,0,1), fmtX=None, fmtY=None ):

    """
    Define graphic axis

    Parameters
    ----------
    ax : objet *axes*
        Objet axes, for instance : ``ax = mpp.subplot(111)``
    fs : *int*
        Police size (*FontSize*)
    clX : *tuple* or *list* or *ndarray*
        Color for abscissa axis with *RGB* format: :math:`(R,V,B) \\in [0,1]^3`
        or with *RGBA* format: :math:`(R,V,B,\\alpha) \\in [0,1]^4` (transparency)
    clY : *tuple* or *list* or *ndarray*
        Same than ``clX``, but for ordinate axis
    fmtX : *string*
        Format for abscissa label, e.g :
        ``fmtX = r$%d$`` for integer with format :math:`\\LaTeX{}`
    fmtY : *string*
        Same than ``fmtX``, but for ordinate axis

    Returns
    -------
    0 : *int*
    """
    #
    from matplotlib.ticker import FormatStrFormatter as fsf
    #
    # Abscissa definition
    for tk in ax.get_xticklabels():
        tk.set_fontsize( fs )
        tk.set_color( clX )
    if (fmtX != None):
        ax.xaxis.set_major_formatter( fsf(fmtX) )
    #
    # Ordinate definition
    for tk in ax.get_yticklabels():
        tk.set_fontsize( fs )
        tk.set_color( clY )
    if (fmtY != None):
        ax.yaxis.set_major_formatter( fsf(fmtY) )
    #
    return None

#end def mefAxes
#------------------------------------------------------------------

#-----------------------------------------------------
# Define graphic axis
#-----------------------------------------------------
def setup_polsection_plot( ax, species ):
    """
    

    Parameters
    ----------
    ax: object *axes*
    species: object *GYSspecies*

    Returns
    -------
    None
    """
    ax.axis('equal')
    ax.axis('off')
    ax.axis((np.amin(species.xx), np.amax(species.xx), \
             np.amin(species.yy), np.amax(species.yy)))
    ax.plot(species.xx[:, 0], species.yy[:, 0], 'k')
    ax.plot(species.xx[:, -1], species.yy[:, -1], 'k')
    ax.plot(species.xx[:, species.i_buffL], species.yy[:, species.i_buffL], 'k--')
    ax.plot(species.xx[:, species.i_buffR], species.yy[:, species.i_buffR], 'k--')
    return None

#end def setup_polsection_plot
#------------------------------------------------------------------
