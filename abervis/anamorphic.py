from dataclasses import dataclass 
import math
import numpy as np

from .lensmodel import W4, TRA, Seidel, Spectrum
        
def z_defocus(f_, A, L, HHx=0.0, HHy=0.0):
    z_x = -f_**2/(HHx+2*f_-L)
    z_y = -f_**2/(HHy+2*f_/A-L)/A**2
    return z_x, z_y
        
def CdCa(D_, z_x, dz_, p_): 
    Cd = 0.5*D_*z_x/(p_+z_x)
    Ca = -Cd*p_*dz_/(p_*z_x+z_x**2-dz_*z_x)
    return Cd, Ca

def WdWa(p_, u_, z_x, dz_):
    Wd = -0.5*u_*z_x/(p_+z_x)
    Wa = -Wd*p_*dz_/(p_*z_x+z_x**2-z_x*dz_)
    return 10e3*Wd, 10e3*Wa # mm->um

# distortion not included as it's better modelled via geometry grid
# as opposed to OPD tilt 
# (which results in a sparse overpadded PSF grid)
def W4_parallel_cyl(
        Py: np.ndarray, Hx=0.0, Hy=0.0, Chi=0.0,
        Wa=0.0, S1y=0.0, S2y=0.0, S3y=0.0, S4y=0.0, D11=0.0,
        Wc120=0.0, Wc140=0.0, Wc220=0.0, Wc240=0.0
        ):
    return (
        (Wc120*Chi + Wc220*Chi*Chi + Wa)*Py**2
        + (Wc140*Chi + Wc240*Chi*Chi - .125*S1y)*Py**4
        - .5*S2y*Hy*Py**3
        - .25*(3*S3y+S4y)*Hy**2*Py**2
        + D11*Hx**2*Py**2
    )

def TRA_y_parallel_cyl(
        Py: np.ndarray, Hx=0.0, Hy=0.0, Chi = 0.0,
        Ca=0.0, S1y=0.0, S2y=0.0, S3y=0.0, S4y=0.0, D11=0.0,
        Cc120y=0.0, Cc140y=0.0, Cc220y=0.0, Cc240y=0.0
        ):
    dy = (
        (Ca + Cc120y*Chi + Cc220y*Chi*Chi)*Py
        + (Cc140y*Chi + Cc240y*Chi*Chi - .5*S1y)*Py**3
        - 1.5*S2y*Hy*Py**2
        - .5*(3*S3y+S4y)*Hy**2*Py
        + 2*D11*Hx**2*Py
    )
    return dy

@dataclass
class AnamorphicParaxial():
    f_: float = 1
    A: float = 2
    HHx: float = 0
    HHy: float = 0
    p_: float = 1
    D_: float = 1

class AnamorphicLens:
    def __init__(self, 
                 parax: AnamorphicParaxial,
                 spectrum: Spectrum ,
                 RSOS_aber: Seidel = None,
                 Y_aber: Seidel = None,
                 ) -> None:
        self.parax = parax
        self.spectrum = spectrum
        # non-localized entrance pupil
        # so we use EXP pos and dia for aperture
        tan_A = self.parax.D_/(2*self.parax.p_)
        self.NA = math.sin(math.atan(tan_A))
        self.RSOS_aber = RSOS_aber
        self.Y_aber = Y_aber
    def _z_defocus(self,L):
        z_x, z_y = z_defocus(
            self.parax.f_,
            self.parax.A,
            L,
            self.parax.HHx,
            self.parax.HHy
        )
        return z_x, z_x - z_y
    def CdCa(self, L):
        z_x, dz_ = self._z_defocus(L)
        return CdCa(self.parax.D_, z_x, dz_, self.parax.p_)
    def WdWa(self, L):
        z_x, dz_ = self._z_defocus(L)
        return WdWa(self.parax.p_, self.NA, z_x, dz_)
    def W(self, Px, Py, L, i_wave=0, Hx=.0, Hy=.0):
        Chi = self.spectrum.Chi(i_wave)
        Wd, Wa = self.WdWa(L)
        return W4(
            Px, Py, Hx, Hy, Chi, Wd,
            *self.RSOS_aber.W_coeffs(self.NA)
            ) + W4_parallel_cyl(
            Py, Hx, Hy, Chi, Wa,
            *self.Y_aber.W_coeffs(self.NA)
            )
    def TRA(self, Px, Py, L, i_wave=0, Hx=.0, Hy=.0):
        Chi = self.spectrum.Chi(i_wave)
        Cd, Ca = self.WdWa(L)
        p_ = self.parax.p_
        tra_x, tra_y = TRA(
            Px, Py, Hx, Hy, Chi, Cd,
            *self.RSOS_aber.TRA_coeffs(self.NA, p_))  
        tra_y += TRA_y_parallel_cyl(
            Py, Hx, Hy, Chi, Ca,
            *self.Y_aber.TRA_coeffs(self.NA, p_))
        return tra_x, tra_y
    def rgb(self, i_wave):
        return self.spectrum.rgb[i_wave]