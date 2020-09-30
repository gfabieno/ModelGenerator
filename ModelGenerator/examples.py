#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Examples using the ModelGenerator class
"""
from ModelGenerator import (ModelGenerator, Property, Stratigraphy, Lithology,
                            Sequence, Deformation)
import argparse
import matplotlib.pyplot as plt
import numpy as np


def random1D():
    """
    Example showing a simple 1D layered model created randomly with a single
    sequence and single property
    """
    gen = ModelGenerator()
    gen.layer_dh_min = 20
    gen.num_layers = 0

    vp = Property(name="vp", vmin=1000, vmax=5000)
    strati = Stratigraphy(defaultprops=[vp])

    gen.dip_max = 0
    gen.max_texture = 0

    gen.num_layers = 0
    gen.layer_num_min = 5
    gen.layer_dh_min = 10

    gen.animated_dataset(strati)


def random2D():
    """
    Example showing a simple 2D layered model created randomly with a single
    sequence and single property
    """

    vp = Property(name="vp", vmin=1000, vmax=5000)
    lith = Lithology(name='vp', properties=[vp])
    deform = Deformation(max_deform_freq=0.1,
                         min_deform_freq=0.0001,
                         amp_max=26,
                         max_deform_nfreq=40,
                         prob_deform_change=0.2)
    sequence = Sequence(lithologies=[lith],
                        ordered=False,
                        deform=deform)
    strati = Stratigraphy(sequences=[sequence])

    gen = ModelGenerator()
    gen.layer_dh_min = 20
    gen.num_layers = 0
    gen.dip_max = 20
    gen.num_layers = 0
    gen.layer_num_min = 5
    gen.layer_dh_min = 10
    gen.animated_dataset(strati)


def fixed2d():
    """
    Example showing a simple 2D layered model with user-assigned thicknesses
    and dips. Contains multiples properties
    """
    name = "Water"
    vp = Property("vp", vmin=1430, vmax=1430)
    vs = Property("vs", vmin=0, vmax=0)
    rho = Property("rho", vmin=1000, vmax=1000)
    q = Property("q", vmin=1000, vmax=1000)
    water = Lithology(name=name, properties=[vp, vs, rho, q])

    name = "Unfrozen sediments"
    vp = Property("vp", vmin=1700, vmax=1700, texture=200)
    vs = Property("vs", vmin=400, vmax=400, texture=150)
    rho = Property("rho", vmin=1900, vmax=1900, texture=150)
    q = Property("q", vmin=50, vmax=50, texture=30)
    unfrozen_sediments = Lithology(name=name, properties=[vp, vs, rho, q])

    name = "Frozen Sands"
    vp = Property("vp", vmin=3700, vmax=3700, texture=200)
    vs = Property("vs", vmin=1600, vmax=1600, texture=250)
    rho = Property("rho", vmin=1900, vmax=1900, texture=150)
    q = Property("q", vmin=60, vmax=60, texture=30)
    frozen_sands = Lithology(name=name, properties=[vp, vs, rho, q])

    deform = Deformation(max_deform_freq=0.02,
                         min_deform_freq=0.0001,
                         amp_max=3,
                         max_deform_nfreq=40,
                         prob_deform_change=0.4)

    sequence = Sequence(lithologies=[water,
                                     unfrozen_sediments,
                                     frozen_sands,
                                     unfrozen_sediments],
                        ordered=True,
                        deform=deform)

    thicks = [25, 25, 100, 50]
    angles = [-0.2, 0.5, 0, 1]
    strati = Stratigraphy(sequences=[sequence])

    gen = ModelGenerator()
    gen.NX = 800
    gen.NZ = 200
    gen.dh = 2.5
    gen.texture_xrange = 3
    gen.texture_zrange = 1.95 * gen.NZ / 2

    props, _, _ = gen.generate_model(strati, thicks=thicks,
                                                 dips=angles)

    fig, axs = plt.subplots(1, 4)
    im = axs[0].imshow(props["vp"], aspect="auto")
    fig.colorbar(im, ax=axs[0], orientation='horizontal')
    im = axs[1].imshow(props["vs"], aspect="auto")
    fig.colorbar(im, ax=axs[1], orientation='horizontal')
    im = axs[2].imshow(props["rho"], aspect="auto")
    fig.colorbar(im, ax=axs[2], orientation='horizontal')
    im = axs[3].imshow(props["q"], aspect="auto")
    fig.colorbar(im, ax=axs[3], orientation='horizontal')
    plt.show()


def random_sequences():
    """
    Example showing how to use multiples sequences to reproduce a usual near
    surface sequence of lithologies.
    """
    name = "unsaturated_sand"
    vp = Property("vp", vmin=300, vmax=500, texture=100)
    vpvs = Property("vpvs", vmin=1.8, vmax=2.5, texture=0.2)
    rho = Property("rho", vmin=1500, vmax=1800, texture=50)
    q = Property("q", vmin=7, vmax=20, texture=4)
    unsaturated_sand = Lithology(name=name, properties=[vp, vpvs, rho, q])

    name = "saturated_sand"
    vp = Property("vp", vmin=1400, vmax=1800, texture=50)
    vpvs = Property("vpvs", vmin=3.5, vmax=12, texture=1)
    rho = Property("rho", vmin=1800, vmax=2200, texture=50)
    q = Property("q", vmin=7, vmax=20, texture=4)
    saturated_sand = Lithology(name=name, properties=[vp, vpvs, rho, q])

    name = "saturated_clay"
    vp = Property("vp", vmin=1500, vmax=1800, texture=50)
    vpvs = Property("vpvs", vmin=6, vmax=20, texture=1)
    rho = Property("rho", vmin=1800, vmax=2200, texture=50)
    q = Property("q", vmin=15, vmax=30, texture=4)
    saturated_clay = Lithology(name=name, properties=[vp, vpvs, rho, q])

    name = "weathered_shale"
    vp = Property("vp", vmin=1950, vmax=2100, texture=50)
    vpvs = Property("vpvs", vmin=2.4, vmax=4.5, texture=1)
    rho = Property("rho", vmin=2000, vmax=2400, texture=50)
    q = Property("q", vmin=15, vmax=30, texture=4)
    weathered_shale = Lithology(name=name, properties=[vp, vpvs, rho, q])

    name = "shale"
    vp = Property("vp", vmin=2000, vmax=2500, texture=20)
    vpvs = Property("vpvs", vmin=2.6, vmax=4.5, texture=1)
    rho = Property("rho", vmin=2000, vmax=2400, texture=50)
    q = Property("q", vmin=30, vmax=60, texture=4)
    shale = Lithology(name=name, properties=[vp, vpvs, rho, q])

    deform = Deformation(max_deform_freq=0.02,
                         min_deform_freq=0.0001,
                         amp_max=8,
                         max_deform_nfreq=40,
                         prob_deform_change=0.1)

    unsat_seq = Sequence(name="unsaturated",
                         lithologies=[unsaturated_sand],
                         thick_max=25,
                         deform=deform)
    sat_seq = Sequence(name="saturated",
                       lithologies=[saturated_clay,
                                    saturated_sand],
                       thick_max=100,
                       deform=deform)
    weathered_seq = Sequence(name="weathered",
                             lithologies=[weathered_shale],
                             thick_max=50, deform=deform)
    roc_seq = Sequence(name="roc", lithologies=[shale], thick_max=99999,
                       deform=deform)

    strati = Stratigraphy(sequences=[unsat_seq,
                                     sat_seq,
                                     weathered_seq,
                                     roc_seq])
    gen = ModelGenerator()
    gen.NX = 800
    gen.NZ = 200
    gen.dh = 2.5
    gen.texture_xrange = 3
    gen.texture_zrange = 1.95 * gen.NZ / 2
    gen.dip_0 = True
    gen.dip_max = 5
    gen.ddip_max = 1
    gen.layer_num_min = 1
    gen.layer_dh_min = 5
    gen.layer_dh_max = 20

    gen.animated_dataset(strati)


def model_with_boundaries():

    vp = Property(name="vp", vmin=1000, vmax=5000)
    lith = Lithology(name='vp', properties=[vp])
    sequence = Sequence(lithologies=[lith, lith, lith, lith], ordered=True)
    strati = Stratigraphy(sequences=[sequence])

    gen = ModelGenerator()
    gen.NX = 100
    gen.NZ = 100
    x = np.arange(gen.NX)

    # Boundaries are drawn from top the top layer to the bottom
    bnd1 = x * 0
    bnd2 = x * 0 + 20
    bnd3 = 10 * np.sin(x/gen.NX * 2 * np.pi * 3) + 40
    # Make boundary 3 appear from the middle
    bnd3[:gen.NX//2] = 9999
    bnd4 = 20 * np.sin(x / gen.NX * 2 * np.pi * 6) + 70
    # Bnd 4 overlaps on layer 2 from 0 to 20
    bnd4[0:20] = 30
    bnds = [bnd1, bnd2, bnd3, bnd4]
    bnds = [b.astype(int) for b in bnds]

    props2d, layerids, _ = gen.generate_model(strati,
                                              boundaries=bnds)

    plt.imshow(props2d["vp"], aspect="auto")
    plt.show()

def model_with_texture_trend():

    vp = Property(name="vp", vmin=1000, vmax=5000, texture=1000)
    lith = Lithology(name='vp', properties=[vp])
    sequence = Sequence(lithologies=[lith, lith, lith, lith], ordered=True)
    strati = Stratigraphy(sequences=[sequence])

    gen = ModelGenerator()
    gen.NX = 100
    gen.NZ = 100
    x = np.arange(gen.NX)

    # Boundaries are drawn from top the top layer to the bottom
    bnd1 = x * 0
    bnd2 = x * 0 + 20
    bnd3 = 10 * np.sin(x/gen.NX * 2 * np.pi * 3) + 40
    bnd4 = 20 * np.sin(x / gen.NX * 2 * np.pi * 6) + 70
    bnds = [bnd1, bnd2, bnd3, bnd4]
    bnds = [b.astype(int) for b in bnds]

    texture_trends = [None, None, x / gen.NX * 50, None]
    gen.texture_xrange = 3
    gen.texture_zrange = 1.95 * gen.NZ / 2

    props2d, layerids, _ = gen.generate_model(strati,
                                              boundaries=bnds,
                                              texture_trends=texture_trends)

    plt.imshow(props2d["vp"], aspect="auto")
    plt.show()


if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="model_with_boundaries",
        help="Name of the example to display"
    )
    # Parse the input for training parameters
    args, unparsed = parser.parse_known_args()

    eval(args.name)()
