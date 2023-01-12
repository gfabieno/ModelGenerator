# ModelGenerator

Generate pseudo-random physical ground models.

This repository is best used in conjunction with [GeoFlow](https://github.com/gfabieno/GeoFlow).

![Sample velocity model](random2D.png)


## Installation

Clone this repository through:

```pip install git+https://github.com/gfabieno/ModelGenerator.git```


## Contents

`ModelGenerator` pseudo-randomly generates ground models through the `ModelGenerator` class. Models are dictionaries of discrete, gridded representations of each of the defined properties. Models are generated by iteratively creating layers given a range of permissible geometrical features and given statiscal contraints for the properties.

The contents of this package are the following:

- `ModelGenerator` class:
  Generates pseudo-random layered model using `generate_model`, given selected lithologies. Holds all parent parameters.

- `Stratigraphy` class:
  A collection of lithostratigraphic sequences. Generates a specific sequence of layers through `build_stratigraphy`. (`Layer` objects hold the properties of a specific layer in a generated model.)

- `Sequence` class:
  An iterable sequence of lithologies.

- `Lithology` iterator:
  An iterable sequence of properties. (`Property` objects statistically describe a single physical property.)

- `Diapir` class:
  A specific lithology. Add a diapir-shaped deformation to layer boundaries.

- `gridded_model` function:
  Generates a gridded representation from a model depicted by a list of layers.

- Various functions and classes that implement specific generation mechanics of a model (`Deformation`, `Faults`, `random_thicks`, `random_dips`, `generate_random_boundaries`, `random_fields`).

Some of the implemented parameters are described in **SIMON Jérome (2023) *3 Écarts de domaine en reconstruction de modèles de vitesse sismique par apprentissage profond: caractérisation de la transférabilité inter-domaines*, Transférabilité et contrôle de qualité en estimation de modèles de vitesse sismique par apprentissage profond, Institut national de la rechercher scientifique**.


## Sample usage

Examples are shown in [ModelGenerator/examples.py](https://github.com/gfabieno/ModelGenerator/blob/master/ModelGenerator/examples.py).
