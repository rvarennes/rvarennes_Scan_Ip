##################################################
# Function and classes for personnal post-treating 
##################################################

import numpy as np 

## Create custom dictionnary class
class mydict(dict):
    def __init__(self, *args, **kwargs):
        self['ls'] = '-'
        super(mydict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    ## Interest of declaring new values here and not in __init__ --> reduce memory usage
    def __getitem__(self, key): 
        # if  key[:2] == 'dr'               : return np.gradient(self[key[2:]], self['rg'], axis=1)
        # if  key[:2] == 'dt'               : return np.gradient(self[key[2:]], self['time'], axis=0)

        if  key[:2] == 'dr'               : return Derivee1(self[key[2:]], self['rg'][1]-self['rg'][0], periodic=False, axis=1)
        if  key[:2] == 'dt'               : return Derivee1(self[key[2:]], self['time'][1]-self['time'][0], periodic=False, axis=0)
        if  key[:4] == 'sqrt'             : return np.sqrt(self[key[4:]])

        if  key     == 'P'                : return -self['As'] * (- self['ns0'] * self['Er'] + 0.5 * self['Zs']**(-1) * self['drPperp'])
        if  key     == 'vorticity'        : return self['drP']
        if  key     == 'Ptransfert'       : return self['drRSpol_vE'] * (-self['Er'])
        if  key     == 'Ptransfert_approx': return self['RSpol_vE'] * (-self['drEr'])
        if  key     == 'Jr'               : return self['As']*(self['Gamma_vE'] + self['Gamma_vD'])
        if  key     == 'ft'               : return np.sqrt(2*self['eps'])
        #if  key     == 'nustar_gia'       : return self['nustar'] / np.sqrt(self['eps'])
        if  key     == 'nutheta_Gianakon' : return (0.452*self['ft']*self['nu_i']) / ( (1+1.03*self['nustar']**(1/2)+0.31*self['nustar'])*(1+0.66*self['nustar']*self['eps']**(3/2)) )
        if  key     == 'RSpol'            : return self['RSpol_vE'] + self['RSpol_vD']
        if  key     == 'RSphi'            : return self['RSphi_vE'] + self['RSphi_vD']
        if  key     == 'Isq_Te_cor'       : return self['Isq_Te'] * self['n']**2
        else: return super().__getitem__(key)
## end of mydict

## Create custom dictionnary class for spectra
class mydict_mnspectra(dict):
    def __init__(self, *args, **kwargs):
        super(mydict_mnspectra, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        self['Nphi'] = len(self['phig'])-1
        self['Ntheta'] = len(self['thetag'])-1
        self['Phithphisq'] = self['Phithphi']**2

        ## Evolution of mn spectra
        tevol_TFPhi2D = np.zeros((len(self['time']),self['Nphi'],self['Ntheta']))
        for it in range(len(self['time'])):
            tevol_TFPhi2D[it,:,:] = np.abs(Fourier2D(self['Phithphisq'][it,0:self['Nphi']+1,0:self['Ntheta']+1],self['phig'],self['thetag'])[0])
        self['TFPhi2D'] = tevol_TFPhi2D

        ## Coordinates in m,n space
        m2d,n2d = Fourier2D(self['Phithphisq'][-1,0:self['Nphi']+1,0:self['Ntheta']+1],self['phig'],self['thetag'])[1:]
        dm2d = (m2d[1]-m2d[0])//2 ; m2d = m2d#-dm2d
        dn2d = (n2d[1]-n2d[0])//2 ; n2d = n2d#-dn2d
        self['m2d'] = m2d 
        self['n2d'] = n2d 
        self['dm2d'] = dm2d
        self['dn2d'] = dn2d

    def estimate_GRlin2D(self,tinit,tend):
        itinit = np.argmin(np.abs(self['time']-tinit))
        itend  = np.argmin(np.abs(self['time']-tend))
        Gammalin2D = np.log(self['TFPhi2D'][itend,:,:]/self['TFPhi2D'][itinit,:,:])/(tend-tinit)
        return Gammalin2D
## end of mydict_mnspectra

## Function to convert a HDF5 file to a custom dictionary
def hdf5_to_dict(filename, cls=mydict):
    import h5py
    """Load a dictionary of arrays and strings as unicode characters from an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        d = {}
        for k in f.keys():
            v = f[k][()]
            if isinstance(v, bytes):
                d[k] = v.decode('utf-8')
            else:
                d[k] = np.array(v)
    return cls(d)
## end of hdf5_to_dict

## 2D Fourier Transform as defined in GYSELA native diagnostics
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
    dkx  = 2. * np.pi / (Lx + dx)
    kx   = np.zeros(nx)
    temp = -dkx * np.r_[1:hnx + 1]
    kx[0:hnx]  = temp[::-1]
    kx[hnx:nx] = dkx * np.r_[0:hnx]

    Ly   = y[ny - 1] - y[0]
    dy   = y[1] - y[0]
    dky  = 2. * np.pi / (Ly + dy)
    ky   = np.zeros(ny)
    temp = -dky * np.r_[1:hny + 1]
    ky[0:hny]  = temp[::-1]
    ky[hny:ny] = dky * np.r_[0:hny]

    TFF = np.zeros((ny, nx), dtype=complex)
    AA  = np.zeros((ny, nx), dtype=complex)
    var = np.conjugate(np.fft.fft2(np.conjugate(F))) / float((nx * ny))

    AA[:, 0:hnx]   = var[:, hnx:nx]
    AA[:, hnx:nx]  = var[:, 0:hnx]
    TFF[0:hny, :]  = AA[hny:ny, :]
    TFF[hny:ny, :] = AA[0:hny, :]

    return TFF, kx, ky
## end of Fourier2D

## 1D Fourier Transform as defined in GYSELA native diagnostics
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

## First derivative as defined in GYSELA native diagnostics
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

## Change hue of a color given in hexadecimal as a string
def change_hue(str_color_hex, factor=0.75):
    import matplotlib.colors as mcolors
    """Change the hue of a color given in hexadecimal.
    """
    # Convert the hexadecimal color to RGB
    rgb = mcolors.hex2color(str_color_hex)
    # Convert the RGB color to HSV
    hsv = mcolors.rgb_to_hsv(rgb)*factor
    # Change the hue and make sure it's between 0 and 1
    hsv = ([min(i,1) for i in (hsv[0], hsv[1], hsv[2])])
    # Convert the HSV color back to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    # Convert the RGB color to hexadecimal
    return mcolors.rgb2hex(rgb)
#end def change_hue