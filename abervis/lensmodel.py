from dataclasses import dataclass, field
import numpy as np

from .color import wavelength_to_rgb

def Chi(waves, i_primary=-1):
    # def: (2.22) p.31 in Rodionov, '82
    if len(waves) < 2:
        return 0 # Chi=0 for monochromatic
    else:
        long = np.max(waves)
        short = np.min(waves)
        if i_primary < 0:
            # only bandwidth specified => use central as primary
            wvl0 = (long+short)/2.0
        else:
            # primary wave specified
            wvl0 = waves[i_primary]
        return [
            (w-wvl0)/(long-short)
            for w in waves
        ]
    
def Wc_coefs(
        NA = 0.0, # = sin^2(omega_A)
        # primary, secondary:
        axcl=0.0, axcl2=0.0, # acial color
        zcl=0.0, zcl2=0.0  # zonal color
):
    NAsq = NA*NA
    return (
        0.25*axcl*NAsq, # Wc120
        0.125*(zcl-axcl)*NAsq, # Wc140
        0.5*axcl2*NAsq, # Wc220
        0.25*(zcl2-axcl2) # Wc240
    )

class Spectrum:
    def __init__(self, waves, i_primary=0):
        self.waves = waves
        self.i_primary = i_primary
        self.primary = waves[i_primary]
        self.Chis = Chi(waves, i_primary)
        self.rgb = [wavelength_to_rgb(w)
                    for w in waves]
    def Chi(self, i_wave):
        return self.Chis[i_wave]
    def rgb(self, i_wave):
        return self.rgb[i_wave]

@dataclass
class Seidel():
    S1: float = .0
    S2: float = .0
    S3: float = .0
    S4: float = .0
    axcl: float = .0
    axcl2: float = .0
    zcl: float = .0
    zcl2: float = .0
    # Wc at NA=1
    color_coefs: list[float] = field(default_factory=list) 
    def __post_init__(self):
        self.color_coefs = Wc_coefs(
            1, self.axcl, self.axcl2, self.zcl, self.zcl2
        )
    def Wc(self, NA):
        return [c*NA for c in self.color_coefs]
    def Cc(self, NA, p_):
        return [-2*p_*c*NA for c in self.color_coefs]
    def TRA_coeffs(self, NA, p_):
        return(
            self.S1, self.S2, self.S3, self.S4,
            *self.Cc(NA, p_)
        )
    def W_coeffs(self, NA):
        return(
            self.S1, self.S2, self.S3, self.S4,
            *self.Wc(NA)
        )

def W4(Px: np.ndarray,Py: np.ndarray, Hx=0.0, Hy=0.0, Chi=0.0,
        Wd=0.0, S1=0.0, S2=0.0, S3=0.0, S4=0.0,
        Wc120=0.0, Wc140=0.0, Wc220=0.0, Wc240=0.0 
        ):
    Psq = Px*Px + Py*Py
    return (
        (Wc120*Chi + Wc220*Chi*Chi + Wd)*Psq
        + (Wc140*Chi + Wc240*Chi*Chi - .125*S1)*Psq*Psq
        - .5*S2*(Hx*Px + Hy*Py)*Psq
        - .25*(3*S3+S4)*(Hx*Hx+Hy*Hy)*Psq
    )

def TRA(Px: np.ndarray, Py: np.ndarray, Hx=0.0, Hy=0.0, Chi=0.0,
        Cd=0.0, S1=0.0, S2=0.0, S3=0.0, S4=0.0,
        Cc120=0.0, Cc140=0.0, Cc220=0.0, Cc240=0.0
        ):
    Psq = Px*Px+Py*Py
    # defocus + longitudinal color:
    defocus = Cd + Cc120*Chi + Cc220*Chi*Chi
    # spherical + spherochromatism:
    sph3 = Cc140*Chi + Cc240*Chi*Chi - .5*S1
    # def: Welford 86', p.111
    dx = (
        defocus*Px
        + sph3*Px*Psq
        - 1.5*S2*Hx*Psq
        - 3*S2*Hy*Px*Py
        - .5*(3*S3+S4)*Hx**2*Px
    )
    dy = (
        defocus*Py
        + sph3*Py*Psq
        - 3*S2*Hx*Px*Py
        - 1.5*S2*Hy*Psq
        - .5*(3*S3+S4)*Hy**2*Py
    )
    return dx, dy