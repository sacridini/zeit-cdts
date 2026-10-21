from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

# zeit's compiled core, in Rust (rust/, see RUST_MIGRATION.md): the extension
# module zeit._core, built from rust/zeit-py with PyO3.
rust_extensions = [
    # debug=False: setuptools-rust otherwise makes in-place builds (build_ext --inplace)
    # debug builds, ten times slower.
    RustExtension('zeit._core', path='rust/zeit-py/Cargo.toml', binding=Binding.PyO3, debug=False),
]

setup(
    name='zeit-cdts',
    version='2.0.0',
    packages=find_packages(include=['zeit', 'zeit.*']),
    package_data={'zeit._plot': ['*.js', '*.css']},
    rust_extensions=rust_extensions,
    zip_safe=False,
)
