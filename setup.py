from __future__ import annotations

import pybind11
from setuptools import Extension, setup

try:
    from jax import ffi
except ImportError:  # JAX 0.4.x
    from jax.extend import ffi


setup(
    ext_modules=[
        Extension(
            "vmex._force_ffi",
            ["vmex/native/force_ffi.cc"],
            include_dirs=[ffi.include_dir(), pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
            optional=True,
        )
    ]
)
