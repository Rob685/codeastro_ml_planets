from setuptools import setup, find_packages, Extension

setup(
    name='orbitize',
    version=get_property('__version__', 'codeastro_ml_planets'),
    description='exo_predict will help you determine how likely you are to find an exoplanet!',
    url='https://github.com/Rob685/codeastro_ml_planets',
    author='Jea Adams, Rob Tejada, Sofia Covarrubias',
    packages=find_packages()
)