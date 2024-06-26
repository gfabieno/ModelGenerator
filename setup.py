
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='ModelGenerator',
      version='0.1.1',
      description='Package layered model generation',
      long_description=readme(),
      author='Gabriel Fabien-Ouellet',
      author_email='gabriel.fabien-ouellet@polymtl.ca',
      license='MIT',
      packages=['ModelGenerator'],
      install_requires=['argparse',
                        'numpy',
                        'h5py',
                        'scipy',
                        'prettytable',
                        'matplotlib'
                        ],
      zip_safe=False)
