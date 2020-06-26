from setuptools import setup, find_packages

setup(
    name='exo_predict',
    version= '1.0.27',
    description='exo_predict will help you determine how likely you are to find an exoplanet!',
    url='https://github.com/Rob685/codeastro_ml_planets',
    author='Jea Adams, Rob Tejada, Sofia Covarrubias',
    python_requires='>=3.6',
    packages=find_packages(),
    package_data={'exo_predict': ['data/*.p']},
    include_package_data=True,
    install_requires=[
        'pandas',
        'sklearn',
        'numpy',
        'xgboost==1.1.1',
        'matplotlib'
    ]
)
