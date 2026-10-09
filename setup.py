import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'zeit._core',
        ['src/main.cpp', 'src/landtrendr.cpp', 'src/ccdc.cpp', 'src/utils.cpp',
         'src/twdtw.cpp', 'src/som.cpp', 'src/phenology_math.cpp',
         'src/phenology_curves.cpp', 'src/phenology.cpp', 'src/mann_kendall.cpp',
         'src/bfast_monitor.cpp', 'src/bfast_lite.cpp', 'src/stl_decompose.cpp',
         'src/bfast.cpp', 'src/snic.cpp', 'src/warp.cpp', 'src/warp_python.cpp',
         'src/whittaker.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'third_party/eigen'
        ],
        language='c++'
    ),
]

def has_flag(compiler, flagname, is_omp=False):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        if is_omp:
            f.write('#include <omp.h>\nint main (int argc, char **argv) { return 0; }')
        else:
            f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-std=c++17')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
            # No fused multiply-add contraction: CCDC reproduces the original's
            # single-precision Fortran GLMnet rounding operation by operation.
            if has_flag(self.compiler, '-ffp-contract=off'):
                opts.append('-ffp-contract=off')

            # OpenMP support
            for ext in self.extensions:
                if sys.platform == 'darwin':
                    # macOS Apple Clang needs specific flags for libomp
                    if has_flag(self.compiler, '-Xpreprocessor', is_omp=True):
                        opts.append('-Xpreprocessor')
                        opts.append('-fopenmp')
                        ext.extra_link_args = ['-lomp']
                else:
                    if has_flag(self.compiler, '-fopenmp', is_omp=True):
                        opts.append('-fopenmp')
                        ext.extra_link_args = ['-fopenmp']
                ext.extra_compile_args = opts

        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            opts.append('/std:c++17')
            opts.append('/openmp')
            for ext in self.extensions:
                ext.extra_compile_args = opts

        build_ext.build_extensions(self)

setup(
    name='zeit-cdts',
    version='0.42.0',
    packages=find_packages(include=['zeit', 'zeit.*']),
    package_data={'zeit._plot': ['*.js', '*.css']},
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.10.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
