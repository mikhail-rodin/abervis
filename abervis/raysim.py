import math
import numpy as np
from scipy.stats import qmc

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