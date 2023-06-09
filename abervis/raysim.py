import math
from functools import reduce
import numpy as np
from scipy import signal
from scipy.stats import qmc
from dataclasses import dataclass
from typing import Iterable

@dataclass
class BitmapSubimage:
    bitmap: np.ndarray
    imsize: Iterable
    Hx: float = .0
    Hy: float = .0
    def __post_init__(self):
        self.pixel_pitch = [h/N for h, N in zip(self.imsize,np.shape(self.bitmap)[:2])]

def poisson_pupil_sampling(
        density: int=16, 
        mc_iterations: int=10, 
        rng: object = None
):
    # creating a stochastic process
    poisson_disk_sp = qmc.PoissonDisk(
        d=2, # 2d disk
        radius=0.5/density, # distance between points
        hypersphere='surface',
        ncandidates=mc_iterations,
        seed=rng
    )
    # sampling from SP -> (x,y) in [0;1] -> [-1;1]
    poisson_points = poisson_disk_sp.fill_space()*2-1
    # radius^2 for masking
    rsq = np.sum(np.square(poisson_points), axis=1)
    # masking
    pupil_samples = poisson_points[rsq <= 1.0]
    return pupil_samples[:,0], pupil_samples[:,1]

def gpsf_pixel_support(tr_x, tr_y, img_pixel_pitch):
    gpsf_sup = [
        [np.min(tr_x), np.max(tr_x)], 
        [np.min(tr_y), np.max(tr_y)]]
    return [
        [math.ceil(abs(sup[0]/pp)), math.ceil(sup[1]/pp)]
        for sup, pp
        in zip(gpsf_sup, img_pixel_pitch)
    ]

def gpsf(
        tr_x, tr_y,
        pixel_support: list[list[int,int],list[int,int]],
        img_pixel_pitch
    ):
    gpsf_edges = [
        np.linspace(
            start = -pix_range[0]*pp,
            stop = pix_range[1]*pp,
            num = pix_range[0]+pix_range[1]
        )
        for pix_range, pp
        in zip(pixel_support, img_pixel_pitch)
    ]
    gpsf, _, _ = np.histogram2d(
        x=tr_x,
        y=tr_y,
        bins = gpsf_edges
    )
    return gpsf.T, gpsf_edges

def _common_support(supports):
    cs = reduce(
        lambda x, y: np.where(np.abs(x)>np.abs(y),x,y),
        [[x for rng in sup for x in rng] for sup in supports])
    return [cs[0:2], cs[2:4]]

def _geom_sim_channel(
    lens: object,
    obj: BitmapSubimage,
    L: float,
    i_wave: int = 0,
    Px: np.ndarray = None, 
    Py: np.ndarray = None, 
):
    if Px is None or Py is None:
        Px, Py = poisson_pupil_sampling()
    tra_x, tra_y = lens.TRA(Px, Py, L, i_wave, obj.Hx, obj.Hy)
    support = gpsf_pixel_support(tra_x, tra_y, obj.pixel_pitch)
    gpsf, _ = gpsf(tra_x, tra_y, support, obj.pixel_pitch)
    img = signal.fftconvolve(obj.bitmap, gpsf, mode='same')
    return np.tensordot(img, lens.rgb(i_wave), axes=0), support

def geom_sim_bw_to_rgb(
    lens: object,
    obj: BitmapSubimage,
    L: float,
    Px: np.ndarray = None, 
    Py: np.ndarray = None, 
):
    waves = lens.spectrum.waves
    imgs = []
    gpsf_supports = []
    for iw in range(len(waves)):
        img, sup = _geom_sim_channel(lens, obj, L, iw, Px, Py)
        imgs.append(img)
        gpsf_supports.append(sup)
    cimg = reduce(np.add, imgs)
    return cimg/np.max(cimg), gpsf_supports