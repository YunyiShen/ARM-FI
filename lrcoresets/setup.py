from distutils.core import setup
import numpy as np

from Cython.Build import cythonize

setup(
    name = 'coresets',
    version='0.1',
    description="Construct coresets for Bayesian logistic regression.",
    author='Jonathan H. Huggins',
    author_email='jhuggins@mit.edu',
    url='https://bitbucket.org/jhhuggins/coresets-for-logistic-regression/',
    packages=['coresets'],
    package_data={'coresets' : ['*.so']},
    install_requires=[
        'Cython >= 0.20.1', 'numpy', 'scipy', 'matplotlib',
        'scikit-learn', 'h5py', 'seaborn', 'nose', 'future'],
    ext_modules = cythonize("coresets/*.pyx"),
    include_dirs = [np.get_include()],
    keywords = ['Bayesian', 'logistic regression', 'scalable inference',
                'coresets'],
    platforms='ALL',
)
