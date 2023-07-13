"""
import distutils.core
import Cython.Build

distutils.core.setup(
    ext_modules = Cython.Build.cythonize("rt_decomp_plotpg_cy.pyx"))
"""
# python setup.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize

mod_names = ["rt_decomp_live_cy",
             "rt_decomp_plotpg_cy"]

for mod_name in mod_names:
    source_file = ["src/"+mod_name+".pyx"]

    extensions = [
        Extension(mod_name, source_file)
    ]
    setup(
        ext_modules=cythonize(extensions, annotate=True),
    )

