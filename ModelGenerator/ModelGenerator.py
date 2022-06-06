#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Class to generate seismic models and labels for training.
"""

import numpy as np
from scipy.signal import gaussian
import h5py as h5
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy


def random_fields(nf, nz, nx, lz=2, lx=2, corr=None):
    """
    Created a random model with bandwidth limited noise.

    @params:
    nf (int): Number of fields to generate
    nz (int): Number of cells in Z
    nx (int): Number of cells in X
    lz (int): High frequency cut-off size in z
    lx (int): High frequency cut-off size in x
    corr (float): Zero-lag correlation between 1 field and subsequent fields
    @returns:

    """

    corrs = [1.0] + [corr for _ in range(nf-1)]

    noise0 = np.random.random([nz, nx])
    noise0 = noise0 - np.mean(noise0)
    noises = []
    for ii in range(nf):
        noisei = np.random.random([nz, nx])
        noisei = noisei - np.mean(noisei)
        noise = corrs[ii] * noise0 + (1.0-corrs[ii]) * noisei
        noise = np.fft.fft2(noise)
        noise[0, :] = 0
        noise[:, 0] = 0

        maskz = gaussian(nz, lz)**2
        maskz = np.roll(maskz, [int(nz / 2), 0])
        if lx > 0:
            maskx = gaussian(nx, lx)**2
            maskx = np.roll(maskx, [int(nx / 2), 0])
            noise *= maskx
        noise = noise * np.reshape(maskz, [-1, 1])

        noise = np.real(np.fft.ifft2(noise))
        noise = noise / np.max(noise)
        if lx == 0:
            noise = np.stack([noise[:, 0] for _ in range(nx)], 1)

        noises.append(noise)

    return noises


def random_thicks(nz, thickmin, thickmax, nmin, nlayer,
                  thick0min=None, thick0max=None):
    """
    Genereate a random sequence of layers with different thicknesses

    :param nz: Number of points in Z of the grid
    :param thickmin: Minimum thickness of a layer in grid point
    :param thickmax: Maximum thickness of a layer in grid point
    :param nmin: Minimum number of layers
    :param nlayer: The number of layers to create. If 0, draws a ramdom
                        number of layers
    :param thick0min: If provided, the first layer thickness is drawn between
                      thick0min and thick0max
    :param thick0max:

    :return: A list containing the thicknesses of the layers

    """

    # Determine the minimum and maximum number of layers
    thickmax = np.min([int(nz / nmin), thickmax])
    if thickmax < thickmin:
        print("warning: maximum number of layers smaller than minimum")
    nlmax = int(nz / thickmin)
    nlmin = int(nz / thickmax)
    if nlayer == 0:
        if nlmin < nlmax:
            nlayer = np.random.randint(nlmin, nlmax)
        else:
            nlayer = nlmin
    else:
        nlayer = int(np.clip(nlayer, nlmin, nlmax))

    amp = (thickmax - thickmin)
    thicks = np.random.uniform(thickmin, thickmax,
                               size=[nlayer]).astype(np.int)

    if thick0max is not None and thick0min is not None:
        thicks[0] = np.random.uniform(thick0min, thick0max)

    tops = np.cumsum(thicks)
    thicks = thicks[tops < nz]

    return thicks


def random_dips(n_dips, dip_max, ddip_max, dip_0=True):
    """
    Generate a random sequence of dips of layers

    :param n_dips: Number of dips to generate
    :param dip_max: Maximum dip
    :param ddip_max: Maximum change of dip
    :param dip_0: If true, the first dip is 0

    :return: A list containing the dips of the thicks
    """

    dips = np.zeros(n_dips)
    if not dip_0:
        dips[1] = np.random.uniform(-dip_max, dip_max)
    for ii in range(2, n_dips):
        dips[ii] = dips[ii - 1] + np.random.uniform(-ddip_max, ddip_max)
        if np.abs(dips[ii]) > dip_max:
            dips[ii] = np.sign(dips[ii]) * dip_max

    return dips


def generate_random_boundaries(nx, layers):
    """
    Generate randomly a boundary for each layer, based on the thickness, dip
    and deformation properties of the sequence to which a layer belong.

    :param nx:
    :param layers:
    :return: layers: The list of layers with the boundary property filled
                     randomly
    """
    top = layers[0].thick
    seq = layers[0].sequence
    de = np.zeros(nx, dtype=np.int)
    for layer in layers[1:]:
        if layer.boundary is None:
            boundary = top
            theta = layer.dip / 360 * 2 * np.pi
            boundary += np.array([int(np.tan(theta) * (jj - nx / 2))
                                  for jj in range(nx)], dtype=np.int)
            if layer.sequence != seq:
                prob = 1
                seq = layer.sequence
            else:
                prob = np.random.rand()
            if seq.deform is not None and prob < seq.deform.prob_deform_change:
                if seq.deform.cumulative:
                    de += seq.deform.create_deformation(nx).astype(np.int)
                else:
                    de = seq.deform.create_deformation(nx)
            boundary += de.astype(np.int)
            boundary = np.clip(boundary, 0, None)
            layer.boundary = boundary
            top += layer.thick

    return layers


def gridded_model(nx, nz, layers, lz, lx, corr):
    """
    Generate a gridded model from a model depicted by a list of Layers objects.
    Add a texture in each layer

    :param nx: Grid size in X
    :param nz: Grid size in Z
    :param layers: A list of Layer objects
    :param lz: The coherence length in z of the random heterogeneities
    :param lx: The coherence length in x of the random heterogeneities
    :param corr: Zero-lag correlation between each property
    :return: A list of 2D grid of the properties and a grid of layer id numbers
    """

    # Generate the 2D model, from top thicks to bottom
    npar = len(layers[0].properties)
    props2d = [np.full([nz, nx], p) for p in layers[0].properties]
    layerids = np.zeros([nz, nx])

    addtext = False
    addtrend = False
    for layer in layers:
        for prop in layer.lithology.properties:
            if lx > 0 and lz > 0 and prop.texture > 0:
                addtext = True
            if np.abs(prop.trend_min) > 1e-6 or np.abs(prop.trend_max) > 1e-6:
                addtrend = True

    if addtext:
        textures = random_fields(npar, 2 * nz, 2 * nx, lz=lz, lx=lx, corr=corr)
        for n in range(npar):
            textamp = layers[0].lithology.properties[n].texture
            if textamp > 0:
                textures[n] = textures[n] / np.max(textures[n])
                props2d[n] += textures[n][:nz, :nx] * textamp

    for layer in layers[1:]:
        trends = [None for _ in range(npar)]
        if addtrend is not None:
            for n in range(npar):
                tmin = layer.lithology.properties[n].trend_min
                tmax = layer.lithology.properties[n].trend_max
                trends[n] = np.random.uniform(tmin, tmax)

        top = np.max(layer.boundary)
        if layer.texture_trend is not None:
            texture_trend = -layer.texture_trend
            texture_trend -= np.min(texture_trend)
            if top + int(np.max(texture_trend)) + nz > 2 * nz:
                top = 2 * nz - int(np.max(texture_trend)) - nz
        else:
            texture_trend = None
        for jj, z in enumerate(layer.boundary):
            for n in range(npar):
                prop = layer.properties[n]
                grad = layer.gradx[n]
                props2d[n][z:, jj] = prop + grad * jj
            layerids[z:, jj] = layer.idnum
            if addtext:
                if layer.texture_trend is not None:
                    b1 = top + z + int(texture_trend[jj])
                else:
                    b1 = top
                b2 = b1 + nz - z
                for n in range(npar):
                    textamp = layer.lithology.properties[n].texture
                    if textamp > 0:
                        props2d[n][z:, jj] += textures[n][b1:b2, jj] * textamp
            if addtrend is not None:
                for n in range(npar):
                    props2d[n][z:, jj] += (trends[n] * np.arange(z, nz))

    # for n in range(npar):
    #     vmin = layers[0].lithology.properties[n].min
    #     vmax = layers[0].lithology.properties[n].max
    #     props2d[n][props2d[n] < vmin] = vmin
    #     props2d[n][props2d[n] > vmax] = vmax

    return props2d, layerids


class Property(object):

    def __init__(self, name="Default", vmin=1000, vmax=5000, texture=0,
                 trend_min=0, trend_max=0, gradx_min=0, gradx_max=0,
                 dzmax=None, filter_decrease=False):
        """
        A Property is used to describe one material property of a Lithology
        object, and provides the maximum and minimum value that can take
        the property within a Lithology.
        For example, a Property could describe the P-wave velocity.

        :param name: Name of the property
        :param vmin: Minimum value of the property
        :param vmax: Maximum value of the property
        :param texture: Maximum percentage of change of random fluctuations
                            within a layer of the property
        :param trend_min: Minimum value of the linear trend in z within a layer
        :param trend_max: Maximum value of the linear trend in z within a layer
        :param gradx_min: Minimum value of the linear trend in x within a layer
        :param gradx_max: Maximum value of the linear trend in x within a layer
        :param dzmax: Maximum change between two consecutive layers with
                      the same lithology
        :param filter_decrease: If true, accept a decrease of this property
                                according to the Sequence accept_decrease value
        """

        self.name = name
        self.min = vmin
        self.max = vmax
        self.texture = texture
        self.trend_min = trend_min
        self.trend_max = trend_max
        self.gradx_min = gradx_min
        self.gradx_max = gradx_max
        self.dzmax = dzmax
        self.filter_decrease = filter_decrease


class Lithology(object):

    def __init__(self, name="Default", properties=None):
        """
        A Lithology is a collection of Property objects.
        For example, a Lithology could be made up of vp and rho for an acoustic
        media and describe unconsolidated sediments seismic properties for a
        marine survey.

        :param name: Name of a lithology
        :param properties: A list of Property objects
        """

        self.name = name
        if properties is None:
            properties = [Property()]
        self.properties = properties
        for prop in properties:
            setattr(self, prop.name, prop)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.properties):
            self.n += 1
            return self.properties[self.n-1]
        else:
            raise StopIteration


class Sequence(object):

    def __init__(self, name="Default", lithologies=None, ordered=False,
                 proportions=None, thick_min=0, thick_max=1e9, nmax=9999,
                 nmin=1, deform=None, skip_prob=0, accept_decrease=1):
        """
        A Sequence object gives a sequence of Lithology objects. It can be
        ordered or random, meaning that when iterated upon, the Sequence object
        will provided either the given lithologies in order, or provide a
        random draw of the lithologies, with a probability of a lithology to
        be drawn given by proportions.

        :param name: Name of the sequence
        :param lithologies: A list of Lithology objects
        :param ordered: If True, the Sequence provides the lithology in the
                        order within lithologies. If False, the Sequence
                        provides random lithologies drawn from the list
        :param proportions: A list of proportions of each lithology in the
                            sequence. Must sum to 1
        :param thick_min: The minimum thickness of the sequence
        :param thick_max: The maximum thickness of the sequence
        :param nmax: Maximum number of lithologies that can be drawn from a
                     sequence, when ordered is False.
        :param nmin: Minimum number of lithologies that can be drawn from a
                     sequence, when ordered is False.
        :param deform: A Deformation object that generate random deformation of
                       a boundary
        :param skip_prob: The probability that this sequence is skipped
        :param accept_decrease: The probability to accept a decrease of a
                               property, for properties with
                               filter_decrease=True
        """

        self.name = name
        if lithologies is None:
            lithologies = [Lithology()]
        self.lithologies = lithologies
        if proportions is not None:
            if np.sum(proportions) - 1.0 > 1e-6:
                raise ValueError("The sum of proportions should be 1")
            if len(proportions) != len(lithologies):
                raise ValueError("Lengths of proportions and lithologies "
                                 "should be equal")
        self.proportions = proportions
        self.ordered = ordered
        if ordered:
            nmax = len(lithologies)
        self.nmax = nmax
        self.nmin = nmin
        self.thick_max = thick_max
        self.thick_min = thick_min
        self.deform = deform
        self.skipprob = skip_prob
        self.accept_decrease = accept_decrease

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.nmax:
            if self.ordered:
                out = self.lithologies[self.n]
            else:
                out = np.random.choice(self.lithologies, p=self.proportions)
            self.n += 1
            return out
        else:
            raise StopIteration


class Layer(object):

    def __init__(self, idnum, thick, dip, sequence, lithology, properties,
                 boundary=None, gradx=None, texture_trend=None):
        """
        A Layer object describes a specific layer within a model, providing a
        description of its lithology, its thickness, dip and the deformation of
        its upper boundary.

        :param idnum: The identification number of the layer
        :param thick: The thickness of the layer
        :param dip: The dip of the layer
        :param sequence: A Sequence object to which the layer belongs
        :param lithology: A Lithology object to which the layer belongs
        :param properties: A list of value of the layer properties
        :param boundary: An array of the position of the top of the layer
        :param gradx: A list of horizontal increase of each property
        """

        self.idnum = idnum
        self.thick = int(thick)
        self.dip = dip
        self.sequence = sequence
        self.lithology = lithology
        self.properties = properties
        names = [prop.name for prop in lithology]
        for name, prop in zip(names, self.properties):
            setattr(self, name, prop)
        if gradx is None:
            gradx = [0 for _ in self.properties]
        self.gradx = gradx
        self.boundary = boundary
        self.texture_trend = texture_trend


class Stratigraphy(object):

    def __init__(self, sequences=None, defaultprops=None):
        """
        A Stratigraphy is made up of a series of Sequences. When building
        a model the layered model will contain the sequences in order.

        :param sequences: A list of Sequence objects
        """
        if sequences is None:
            litho = Lithology(properties=defaultprops)
            sequences = [Sequence(lithologies=[litho])]

        self.sequences = sequences
        self.layers = None

    def properties(self):
        """
        Summarize the properties in Stratigraphy

        :return:
            props: A dict containing all properties contained in Stratigraphy
                   with minimum and maximum values {pro_name: [min, max]
        """
        props = {p.name: [9999, 0]
                 for p in self.sequences[0].lithologies[0]}
        for seq in self.sequences:
            for lith in seq:
                for p in lith.properties:
                    if props[p.name][0] > p.min:
                        props[p.name][0] = p.min
                    if props[p.name][1] < p.max:
                        props[p.name][1] = p.max
        return props

    def build_stratigraphy(self, thicks, dips, gradxs=None):
        """
        Generate a sequence of Layer object that provides properties of each
        layer in a stratigraphic column.

        :param thicks: A list of layer thicknesses
        :param dips: A list of layer dips
        :param gradxs: A list of the linear trend of each property in each layer
                       If None, no trend in x is added and if "random", create
                       random gradients in each layer.
        :return:
        """

        layers = []
        seqid = 0
        seqid0 = 0
        seqthick = 0
        sequences = [s for s in self.sequences if s.skipprob < np.random.rand()]
        seqiter = iter(sequences[0])

        sthicks_min = [np.random.randint(s.thick_min, s.thick_max)
                       for s in sequences]
        sthicks_max = [np.random.randint(smin, s.thick_max)
                       for s, smin in zip(sequences, sthicks_min)]
        sthicks_min[-1] = sthicks_max[-1] = 1e09

        seq = sequences[0]
        lith = None
        properties = [0.0 for _ in sequences[0].lithologies[0]]
        seq_nlay = 0
        for ii, (t, di) in enumerate(zip(thicks, dips)):
            seqthick0 = seqthick
            seqthick += t
            if seq_nlay >= seq.nmin and (seqthick0 > sthicks_min[seqid]
                                         or seqthick >= sthicks_max[seqid]):
                seq_nlay = 0
                if seqid < len(sequences) - 1:
                    seqid0 = seqid
                    seqid += 1
                    seqiter = iter(sequences[seqid])
                    seq = sequences[seqid]
            seq_nlay += 1
            lith0 = lith
            lith = next(seqiter)

            if gradxs is None:
                gradx = None
            elif gradxs == "random":
                gradx = [0 for _ in lith]
                for n, prop in enumerate(lith):
                    gradxs[n] = np.random.rand(prop.gradx_min, prop.gradx_max)
            else:
                gradx = gradxs[ii]

            for jj, prop in enumerate(lith):
                if prop.dzmax is not None and lith0 is not None:
                    minval = properties[jj] - prop.dzmax
                    maxval = properties[jj] + prop.dzmax
                    if minval < prop.min:
                        minval = prop.min
                    if maxval > prop.max:
                        maxval = prop.max
                else:
                    minval = prop.min
                    maxval = prop.max
                if seqid == seqid0 and seq.accept_decrease < np.random.rand():
                    if prop.filter_decrease:
                        if prop.min <= properties[jj]:
                            minval = properties[jj]
                        if maxval < minval:
                            maxval = minval

                properties[jj] = np.random.uniform(minval, maxval)

            layers.append(Layer(ii, t, di, seq, lith, gradx=gradx,
                                properties=copy.copy(properties)))

        self.layers = layers

        return layers

    def summary(self):
        x = PrettyTable()
        x.add_column("Layer no", [la.idnum for la in self.layers])
        x.add_column("Sequence",  [la.sequence.name for la in self.layers])
        x.add_column("Lithology",  [la.lithology.name for la in self.layers])
        x.add_column("Thickness",  [la.thick for la in self.layers])
        x.add_column("Dip",  [la.dip for la in self.layers])
        for ii in range(len(self.layers[0].lithology.properties)):
            x.add_column(self.layers[0].lithology.properties[ii].name,
                         [la.properties[ii] for la in self.layers])
        print(x)


class Deformation:

    def __init__(self, max_deform_freq=0, min_deform_freq=0, amp_max=0,
                 max_deform_nfreq=20, prob_deform_change=0.3, cumulative=False):
        """
        Create random deformations of a boundary with random harmonic functions

        :param max_deform_freq: Maximum frequency of the harmonic components
        :param min_deform_freq: Minimum frequency of the harmonic components
        :param amp_max: Maximum amplitude of the deformation
        :param max_deform_nfreq: Number of frequencies
        :param cumulative:      Bool, if True, deformation of consecutive layers
                                are added together (are correlated).
        """
        self.max_deform_freq = max_deform_freq
        self.min_deform_freq = min_deform_freq
        self.amp_max = amp_max
        self.max_deform_nfreq = max_deform_nfreq
        self.prob_deform_change = prob_deform_change
        self.cumulative = cumulative

    def create_deformation(self, nx):
        """
        Create random deformations of a boundary with random harmonic functions
        :param nx: Number of points of the boundary

        :return:
        An array containing the deformation function
        """
        x = np.arange(0, nx)
        deform = np.zeros(nx)
        if self.amp_max > 0 and self.max_deform_freq > 0:
            nfreqs = np.random.randint(self.max_deform_nfreq)
            vmin = np.log(self.max_deform_freq)
            vmax = np.log(self.min_deform_freq)
            freqs = np.exp(np.random.uniform(vmin, vmax, size=nfreqs))
            phases = np.random.rand(nfreqs) * np.pi * 2
            amps = np.random.rand(nfreqs)
            for ii in range(nfreqs):
                deform += amps[ii] * np.sin(freqs[ii] * x + phases[ii])

            ddeform = np.max(np.abs(deform))
            if ddeform > 0:
                deform = deform / ddeform * self.amp_max * np.random.rand()

        return deform


class Faults:

    def __init__(self, dip_min=0, dip_max=0, displ_min=0, displ_max=0, dh=10.0,
                 x_lim=None, y_lim=None, nmax=1, prob=0):
        """
        Create random faults in a 2D gridded model.

        :param dip_min: Minimum dip, as measured in degrees from the
                        horizontal axis.
        :param dip_max: Maximum dip, as measured in degrees from the
                        horizontal axis.
        :param displ_min: Minimum absolute displacement, in meters. A positive
                          displacement brings the top layer upwards.
        :param displ_max: Maximum absolute displacement, in meters. A positive
                          displacement brings the top layer upwards.
        :param dh: Grid cell size, in meters.
        :param x_lim: Bounds `[x_min, x_max]` of the fault origin location.
                      Defaults to the model's boundaries.
        :param y_lim: Bounds `[y_min, y_max]` of the fault origin location.
                      Defaults to the model's boundaries. `y` is measured from
                      the surface.
        :param nmax: Maximum quantity of faults.
        :param prob: Either the scalar probability of having at least one fault
                     or a list of probabilities for each quantity of faults
                     in `range(1, nmax+1)`. In either case, the remaining
                     probability is associated to not having any fault.
        """
        self.dip_min = np.deg2rad(dip_min)
        self.dip_max = np.deg2rad(dip_max)
        self.displ_min = displ_min
        self.displ_max = displ_max
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.dh = dh
        self.nmax = nmax
        if isinstance(prob, list):
            assert len(prob) == nmax
            prob = [1-sum(prob), *prob]
        else:
            prob = [1 - prob, *[prob/nmax]*nmax]
        self.prob = prob

    def add_faults(self, props2d, layerids):
        n = np.random.choice(self.nmax+1, p=self.prob)
        for _ in range(n):
            props2d, layerids = self.add_fault(props2d, layerids)
        return props2d, layerids

    def add_fault(self, props2d, layerids):
        dip = np.random.uniform(self.dip_min, self.dip_max)
        dip *= np.random.choice([-1, 1])
        displ = np.random.uniform(self.displ_min, self.displ_max)
        displ /= self.dh
        vdispl = displ * np.sin(abs(dip))
        vdispl = int(round(vdispl))
        if vdispl == 0:
            return props2d, layerids

        x_min, x_max = self.x_lim or (0, layerids.shape[1])
        y_min, y_max = self.y_lim or (0, layerids.shape[0])
        y = layerids.shape[0] - np.random.randint(y_min, y_max)
        x = np.random.randint(x_min, x_max)

        grid_idx = np.meshgrid(*(np.arange(s) for s in layerids.shape[::-1]))
        grid_idx = np.array(grid_idx).reshape([2, -1]).T
        grid_idx -= [x, y]
        line_idx = np.cos(dip), np.sin(dip)
        is_over = np.cross(line_idx, grid_idx) < 0
        is_over = is_over.reshape(layerids.shape)

        arrays = [*props2d, layerids]
        for i, arr in enumerate(arrays):
            if vdispl > 0:
                displ_arr = np.pad(arr, ((0, vdispl), (0, 0)), mode='edge')
                displ_arr = displ_arr[vdispl:]
            else:
                displ_arr = np.pad(arr, ((-vdispl, 0), (0, 0)), mode='edge')
                displ_arr = displ_arr[:vdispl]

            upper, lower = displ_arr, arr
            arrays[i] = np.where(is_over, upper, lower)

        *props2d, layerids = arrays
        return props2d, layerids


class ModelGenerator:
    """
    Generate a layered model with the generate_model method.
    This class can read and write to a file the parameters needed to generate
    random models
    """

    def __init__(self):

        # Number of grid cells in X direction.
        self.NX = 256
        # Number of grid cells in Z direction.
        self.NZ = 256
        # Grid spacing in X, Y, Z directions (in meters).
        self.dh = 10.0

        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_min = 50
        # Minimum thickness of a layer (in grid cells).
        self.layer_dh_max = 1e9
        # Minimum number of layers.
        self.layer_num_min = 5
        # Fix the number of layers if not 0.
        self.num_layers = 0

        # If true, first layer dip is 0.
        self.dip_0 = True
        # Maximum dip of a layer.
        self.dip_max = 0
        # Maximum dip difference between two adjacent layers.
        self.ddip_max = 5

        # Change between two layers.
        # Add random noise two a layer (% or velocity).
        self.max_texture = 0
        # Range of the filter in x for texture creation.
        self.texture_xrange = 0
        # Range of the filter in z for texture creation.
        self.texture_zrange = 0
        # Zero-lag correlation between parameters, same for each
        self.corr = 0.6

        # Minimum fault dip.
        self.fault_dip_min = 0
        # Maximum fault dip.
        self.fault_dip_max = 0
        # Minimum fault displacement.
        self.fault_displ_min = 0
        # Maximum fault displacement.
        self.fault_displ_max = 0
        # Bounds of the fault origin location.
        self.fault_x_lim = [0, self.NX]
        self.fault_y_lim = [0, self.NZ]
        # Maximum quantity of faults.
        self.fault_nmax = 1
        # Probability of having faults.
        self.fault_prob = 0

        self.thick0min = None
        self.thick0max = None
        self.layers = None

    def save_parameters_to_disk(self, filename):
        """
        Save all parameters to disk

        @params:
        filename (str) :  name of the file for saving parameters

        @returns:

        """
        with h5.File(filename, 'w') as file:
            for item in self.__dict__:
                file.create_dataset(item, data=self.__dict__[item])

    def read_parameters_from_disk(self, filename):
        """
        Read all parameters from a file

        @params:
        filename (str) :  name of the file containing parameters

        @returns:

        """
        with h5.File(filename, 'r') as file:
            for item in self.__dict__:
                try:
                    self.__dict__[item] = file[item][()]
                except KeyError:
                    pass

    def generate_model(self, stratigraphy, thicks=None, dips=None,
                       boundaries=None, gradxs=None, texture_trends=None,
                       seed=None):
        """

        :param stratigraphy: A stratigraphy object
        :param thicks: A list of layer thicknesses. If not provided, create
                       random thicknesses. See ModelParameters for variables
                       controlling the random generation process.
        :param dips: A list of layer dips. If not provided, create
                       random dips.
        :param boundaries: A list of arrays containing the position of the top
                           of the layers. If none, generated randomly
        :param gradxs: A list of the linear trend of each property in each layer
                       If None, no trend in x is added and if "random", create
                       random gradients in each layer.
        :param texture_trends: A list of the of array depicting the alignment of
                              the texture within a layer. If None, will follow
                              the top boundary of the layer.
        :param seed: A seed for random generators

        :return:
                props2d: A list of 2D property arrays
                layerids: A 2D array containing layer ids
                layers: A list of Layer objects
        """

        if seed is not None:
            np.random.seed(seed)

        if stratigraphy is None:
            stratigraphy = Stratigraphy()
        if thicks is None:
            if boundaries is None:
                thicks = random_thicks(self.NZ, self.layer_dh_min,
                                       self.layer_dh_max,
                                       self.layer_num_min, self.num_layers,
                                       thick0min=self.thick0min,
                                       thick0max=self.thick0max)
            else:
                thicks = [0 for _ in range(len(boundaries))]
        if dips is None:
            dips = random_dips(len(thicks), self.dip_max,
                               self.ddip_max, dip_0=self.dip_0)

        layers = stratigraphy.build_stratigraphy(thicks, dips, gradxs=gradxs)
        if boundaries is None:
            layers = generate_random_boundaries(self.NX, layers)
        else:
            for ii, layer in enumerate(layers):
                layer.boundary = boundaries[ii]
        if texture_trends is not None:
            for ii, layer in enumerate(layers):
                layer.texture_trend = texture_trends[ii]

        props2d, layerids = gridded_model(self.NX, self.NZ, layers,
                                          self.texture_zrange,
                                          self.texture_xrange,
                                          self.corr)
        faults = Faults(dip_min=self.fault_dip_min, dip_max=self.fault_dip_max,
                        displ_min=self.fault_displ_min,
                        displ_max=self.fault_displ_max, dh=self.dh,
                        x_lim=self.fault_x_lim, y_lim=self.fault_y_lim,
                        nmax=self.fault_nmax, prob=self.fault_prob)
        props2d, layerids = faults.add_faults(props2d, layerids)

        names = [prop.name for prop in layers[0].lithology]
        propdict = {name: prop for name, prop in zip(names, props2d)}
        return propdict, layerids, layers

    def plot_model(self, props2d, layers, animated=False, figsize=(16, 8)):
        """
        Plot the properties of a generated gridded model

        :param props2d: The dictionary of properties from the output of
                        generate_model
        :param layers:  A list of layers from the output of  generate_model
        :param animated: It true, the plot can be animated
        :param figsize: A tuple providing the size of the figure to create

        :return: ims: a list of pyplot images
                 fig: A Figure object
        """
        names = list(props2d.keys())
        minmax = {name: [np.inf, -np.inf] for name in names}
        for layer in layers:
            for prop in layer.lithology:
                if prop.name in minmax:
                    if minmax[prop.name][0] > prop.min:
                        minmax[prop.name][0] = prop.min
                    if minmax[prop.name][1] < prop.max:
                        minmax[prop.name][1] = prop.max
        for name in names:
            if minmax[name][0] is np.inf:
                minmax[name] = [np.min(props2d[name]) / 10,
                                np.min(props2d[name]) * 10]

        fig, axs = plt.subplots(1, len(names), figsize=figsize, squeeze=False)
        axs = axs.flatten()
        ims = [axs[ii].imshow(props2d[name], animated=animated, aspect='auto',
                              cmap='inferno', vmin=minmax[name][0],
                              vmax=minmax[name][1])
               for ii, name in enumerate(names)]

        for ii, ax in enumerate(axs):
            ax.set_title(names[ii])
            plt.colorbar(ims[ii], ax=ax, orientation="horizontal", pad=0.16,
                         fraction=0.15)
        plt.tight_layout()

        return fig, ims

    def animated_dataset(self, *args, filename=None, nframes=1000, **kwargs):
        """
        Produces an animation of a dataset, showing the input data, and the
        different labels for each example.

        @params:
        phase (str): Which dataset: either train, test or validate
        """

        props2d, _, layers = self.generate_model(*args, **kwargs)
        names = list(props2d.keys())
        fig, ims = self.plot_model(props2d, layers, animated=True)

        def init():
            for im, name in zip(ims, names):
                im.set_array(props2d[name])
            return ims

        def animate(t):
            props2d, _, layers = self.generate_model(*args, **kwargs)
            for im, name in zip(ims, names):
                im.set_array(props2d[name])
            return ims

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=nframes, interval=3000, blit=True,
                                       repeat=True)
        if filename:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=1, metadata=dict(artist='ModelGenerator'),
                            bitrate=1800)
            anim.save(filename + ".mp4", writer=writer)

        plt.show()


if __name__ == '__main__':
    gen = ModelGenerator()
    stratigraphy = Stratigraphy()
    gen.animated_dataset(stratigraphy)
