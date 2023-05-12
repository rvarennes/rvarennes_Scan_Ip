##################################################
# Function and classes for personnal post-treating 
##################################################

import numpy as np 

## Create custom dictionnary class
class mydict(dict):
    def __init__(self, *args, **kwargs):
        super(mydict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    ## Interest of declaring new values here and not in __init__ --> reduce memory usage
    def __getitem__(self, key): 
        if  key[:2] == 'dr'               : return np.gradient(self[key[2:]], self['rg'], axis=1)
        if  key[:2] == 'dt'               : return np.gradient(self[key[2:]], self['time'], axis=0)
        if  key     == 'P'                : return -self['As'] * (- self['ns0'] * self['Er'] + 0.5 * self['Zs']**(-1) * self['drPperp'])
        if  key     == 'vorticity'        : return self['drP']
        if  key     == 'Ptransfert'       : return self['drRSpol_vE'] * (-self['Er'])
        if  key     == 'Ptransfert_approx': return self['RSpol_vE'] * (-self['drEr'])
        if  key     == 'Jr'               : return self['As']*(self['Gamma_vE'] + self['Gamma_vD'])
        else: return super().__getitem__(key)
## end of mydict

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