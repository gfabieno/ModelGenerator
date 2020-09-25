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
    thicks = (thickmin + np.random.rand(nlayer) * amp).astype(np.int)

    if thick0max is not None and thick0min is not None:
        thicks[0] = thick0min + np.random.rand() * (thick0max - thick0min)

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
        dips[1] = -dip_max + np.random.rand() * 2 * dip_max
    for ii in range(2, n_dips):
        dips[ii] = (
            dips[ii - 1] + (2.0*np.random.rand()-1.) * ddip_max
        )
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
    props2d = [np.zeros([nz, nx]) + p for p in layers[0].properties]
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
                textures[n] = textures[n] / np.max(textures[n]) * textamp
                props2d[n] += textures[n][:nz, :nx]

    top = layers[0].thick
    for layer in layers[1:]:

        if addtext:
            for n in range(npar):
                textamp = layer.lithology.properties[n].texture
                if textamp > 0:
                    textures[n] = textures[n] / np.max(textures[n]) * textamp

        trends = [None for _ in range(npar)]
        if addtrend is not None:
            for n in range(npar):
                tmin = layer.lithology.properties[n].trend_min
                tmax = layer.lithology.properties[n].trend_max
                trends[n] = tmin + np.random.rand() * (tmax - tmin)

        for jj, z in enumerate(layer.boundary):
            for n in range(npar):
                prop = layer.properties[n]
                grad = layer.gradx[n]
                props2d[n][z:, jj] = prop + grad * jj
            layerids[z:, jj] = layer.idnum
            if addtext:
                for n in range(npar):
                    textamp = layer.lithology.properties[n].texture
                    if textamp > 0:
                        props2d[n][z:, jj] += textures[n][top:nz + top - z, jj]
            if addtrend is not None:
                for n in range(npar):
                    props2d[n][z:, jj] += (trends[n] * np.arange(z, nz))
        top += layer.thick

    # for n in range(npar):
    #     vmin = layers[0].lithology.properties[n].min
    #     vmax = layers[0].lithology.properties[n].max
    #     props2d[n][props2d[n] < vmin] = vmin
    #     props2d[n][props2d[n] > vmax] = vmax

    return props2d, layerids


class Property(object):

    def __init__(self, name="Default", vmin=1000, vmax=5000, texture=0,
                 trend_min=0, trend_max=0, gradx_min=0, gradx_max=0,
                 dzmax=None):
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
                 proportions=None, thick_min=0, thick_max=1e12, nmax=9999,
                 deform=None):
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
        :param deform: A Deformation object that generate random deformation of
                        a boundary
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

        self.thick_max = thick_max
        self.thick_min = thick_min
        self.deform = deform

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.nmax:
            if self.ordered:
                out = self.lithologies[self.n]
            else:
                if self.proportions is not None:
                    a = np.random.rand()
                    flo = 0
                    for ii, b in enumerate(self.proportions):
                        flo += b
                        if a <= b:
                            break
                    out = self.lithologies[ii]
                else:
                    out = np.random.choice(self.lithologies)
            self.n += 1
            return out
        else:
            raise StopIteration


class Layer(object):

    def __init__(self, idnum, thick, dip, sequence, lithology, properties,
                       boundary=None, gradx=None):
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
        seqthick = 0
        seqiter = iter(self.sequences[0])
        sthicks = [s.thick_min + np.random.rand() * (s.thick_max - s.thick_min)
                   for s in self.sequences]
        sthicks[-1] = 1e09

        seq = self.sequences[0]
        lith = None
        properties = [0.0 for _ in self.sequences[0].lithologies[0]]
        for ii, (t, di) in enumerate(zip(thicks, dips)):
            seqthick += t
            if seqthick > sthicks[seqid]:
                if seqid < len(self.sequences) - 1:
                    seqid += 1
                    seqiter = iter(self.sequences[seqid])
                    seq = self.sequences[seqid]

            lith0 = lith
            lith = next(seqiter)
            if gradxs is None:
                gradx = None
            elif gradxs is "random":
                gradx = [0 for _ in lith]
                for n, prop in enumerate(lith):
                    amp = prop.gradx_max-prop.gradx_min
                    gradxs[n] = prop.gradx_min + np.random.rand() * amp
            else:
                gradx = gradxs[ii]

            for jj, prop in enumerate(lith):
                if prop.dzmax is not None and lith0 == lith:
                    amp = 2 * prop.dzmax
                    minval = properties[jj] - prop.dzmax
                else:
                    minval = prop.min
                    amp = (prop.max - prop.min)
                properties[jj] = minval + np.random.rand() * amp

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
                 max_deform_nfreq=20, prob_deform_change=0.3):
        """
        Create random deformations of a boundary with random harmonic functions

        :param max_deform_freq: Maximum frequency of the harmonic components
        :param min_deform_freq: Minimum frequency of the harmonic components
        :param amp_max: Maximum amplitude of the deformation
        :param max_deform_nfreq: Number of frequencies
        """
        self.max_deform_freq = max_deform_freq
        self.min_deform_freq = min_deform_freq
        self.amp_max = amp_max
        self.max_deform_nfreq = max_deform_nfreq
        self.prob_deform_change = prob_deform_change

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
            vmin = self.min_deform_freq
            amp = (self.max_deform_freq - self.min_deform_freq)
            freqs = vmin + amp * np.random.rand(nfreqs)
            phases = np.random.rand(nfreqs) * np.pi * 2
            amps = np.random.rand(nfreqs)
            for ii in range(nfreqs):
                deform += amps[ii] * np.sin(freqs[ii] * x + phases[ii])

            ddeform = np.max(np.abs(deform))
            if ddeform > 0:
                deform = deform / ddeform * self.amp_max * np.random.rand()

        return deform


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
                       boundaries=None, gradxs=None, seed=None):
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

        props2d, layerids = gridded_model(self.NX, self.NZ, layers,
                                          self.texture_zrange,
                                          self.texture_xrange,
                                          self.corr)

        names = [prop.name for prop in layers[0].lithology]
        propdict = {name: prop for name, prop in zip(names, props2d)}
        return propdict, layerids, layers

    def animated_dataset(self, *args, **kwargs):
        """
        Produces an animation of a dataset, showing the input data, and the
        different labels for each example.

        @params:
        phase (str): Which dataset: either train, test or validate
        """

        toplots, _, layers = self.generate_model(*args, **kwargs)
        # names = [prop.name for prop in layers[0].lithology]
        names = list(toplots.keys())
        minmax = {name: [np.inf, -np.inf] for name in names}
        for layer in layers:
            for prop in layer.lithology:
                if minmax[prop.name][0] > prop.min:
                    minmax[prop.name][0] = prop.min
                if minmax[prop.name][1] < prop.max:
                    minmax[prop.name][1] = prop.max
        for name in names:
            if minmax[name][0] is np.inf:
                minmax[name] = [np.min(toplots[name]) / 10,
                                np.min(toplots[name]) * 10]

        fig, axs = plt.subplots(1, len(names), figsize=[16, 8], squeeze=False)
        axs = axs.flatten()
        ims = [axs[ii].imshow(toplots[name], animated=True, aspect='auto',
                              cmap='inferno', vmin=minmax[name][0],
                              vmax=minmax[name][1])
               for ii, name in enumerate(names)]

        for ii, ax in enumerate(axs):
            ax.set_title(names[ii])
            plt.colorbar(ims[ii], ax=ax, orientation="horizontal", pad=0.05,
                         fraction=0.2)
        plt.tight_layout()

        def init():
            for im, name in zip(ims, names):
                im.set_array(toplots[name])
            return ims

        def animate(t):
            toplots, _, layers = self.generate_model(*args, **kwargs)
            for im, name in zip(ims, names):
                im.set_array(toplots[name])
            return ims

        _ = animation.FuncAnimation(fig, animate, init_func=init, frames=1000,
                                    interval=3000, blit=True, repeat=True)
        plt.show()


