
# coding: utf-8
import os
import sys
from glob import glob
import shutil
import subprocess


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable, os.path.join(cwd, 'cythonize.py'), '.'],
                        cwd = cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    generate_cython()
    config = Configuration('', parent_package, top_path)
    libraries = []
    # TODO: switch from Og (debugging) to O3 (optimised)
    my_compile_args = ['-Og', '-march=native']
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension('ball_tree',
                         sources=['ball_tree.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args = my_compile_args)

    config.add_extension('kd_tree',
                         sources=['kd_tree.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args = my_compile_args)

    config.add_extension('dist_metrics',
                         sources=['dist_metrics.c'],
                         include_dirs=[numpy.get_include(),
                                       os.path.join(numpy.get_include(),
                                                    'numpy')],
                         libraries=libraries,
                         extra_compile_args = my_compile_args)

    config.add_extension('typedefs',
                         sources=['typedefs.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args = my_compile_args)

    # config.add_extension('distance',
    #                      sources=['distance.c'],
    #                      include_dirs=[numpy.get_include()],
    #                      libraries=libraries,
    #                      extra_compile_args = my_compile_args)

    config.add_extension('faster_sandwich_filling',
                         sources=['faster_sandwich_filling.c'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration = configuration)
    # hack to copy build extension into this directory so we can use it without
    # setting paths.
    for src in glob("build/lib*/*"):
        shutil.copy(src, ".")
