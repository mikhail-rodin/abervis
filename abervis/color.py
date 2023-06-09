import numpy as np
import colour

def wavelength_to_rgb(wvl_nm):
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    model = colour.models.RGB_COLOURSPACE_sRGB
    RGB = colour.XYZ_to_RGB(
        XYZ=colour.wavelength_to_XYZ(wvl_nm, cmfs),
        illuminant_XYZ = np.array([0.34570, 0.35850]),
        illuminant_RGB = np.array([0.31270, 0.32900]),
        matrix_XYZ_to_RGB=model.matrix_XYZ_to_RGB
    )
    # desaturate RGB to move out-of-gamut colors inside the RGB triagle
    d_white = -np.min(np.append(RGB,0))
    RGB8bit = 255*colour.algebra.normalise_maximum(RGB + d_white, clip=True)
    return RGB8bit.astype(np.uint8)

def wvl_to_rgb_matrix(waves):
    return np.column_stack(
        [wavelength_to_rgb(w) for w in waves]
    )