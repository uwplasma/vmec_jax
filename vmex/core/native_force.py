"""Opt-in native force projection with an exact portable derivative rule."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

try:
    from jax import ffi
except ImportError:  # JAX 0.4.x
    from jax.extend import ffi  # type: ignore[no-redef]

from .transforms import (
    SpectralForce,
    _radial_masks,
    _tomnsp_theta_stage,
    _tomnsp_zeta_stage,
)

try:
    from vmex import _force_ffi
except ImportError:  # optional extension: the pure-JAX package remains valid
    _force_ffi = None
else:
    try:
        for _name, _target in _force_ffi.registrations().items():
            ffi.register_ffi_target(_name, _target, platform="cpu")
    except Exception:  # incompatible optional binary/JAX FFI API
        _force_ffi = None


_NAMES = (
    "force_R", "force_R_du", "force_R_dv", "force_Z", "force_Z_du",
    "force_Z_dv", "force_lambda_du", "force_lambda_dv",
    "constraint_R", "constraint_Z",
)


def native_force_available() -> bool:
    """Whether the optional CPU FFI extension was built."""
    return _force_ffi is not None


def require_native_cpu(device, resolution) -> None:
    """Reject an explicit/effective accelerator before tracing the CPU FFI."""
    from .device import _placement_device

    target = _placement_device(device, resolution)
    if target is None:
        target = jax.config.jax_default_device or jax.devices()[0]
    if target.platform != "cpu":
        raise ValueError(
            "force_backend='native' is CPU-only; use device='cpu' or "
            "force_backend='jax' on accelerators"
        )


def _reference(packed, cosmui, sinmui, cosmumi, sinmumi, cosnv, sinnv,
               cosnvn, sinnvn, *, mpol, ntor, include_edge, asym):
    class Tables:
        pass

    trig = Tables()
    trig.ntheta2 = int(cosmui.shape[0])
    trig.cosmui, trig.sinmui = cosmui, sinmui
    trig.cosmumi, trig.sinmumi = cosmumi, sinmumi
    trig.cosnv, trig.sinnv = cosnv, sinnv
    trig.cosnvn, trig.sinnvn = cosnvn, sinnvn
    kernels = {
        f"{name}_{parity}": packed[2 * i + p]
        for i, name in enumerate(_NAMES)
        for p, parity in enumerate(("even", "odd"))
    }
    work = _tomnsp_theta_stage(kernels, mpol=mpol, trig=trig)
    if asym:
        cos_pairs = ((2, 3), (4, 5), (8, 9))
        sin_pairs = ((0, 1), (6, 7), (10, 11))
    else:
        cos_pairs = ((0, 1), (6, 7), (10, 11))
        sin_pairs = ((2, 3), (4, 5), (8, 9))
    cos_out, sin_out = _tomnsp_zeta_stage(
        work, ntor=ntor, trig=trig, cos_pairs=cos_pairs, sin_pairs=sin_pairs
    )
    if sin_out is None:
        sin_out = tuple(jnp.zeros_like(value) for value in cos_out)
    ns = int(packed.shape[1])
    mask_rz, mask_l = _radial_masks(ns, mpol, include_edge, packed.dtype)
    return jnp.stack((
        cos_out[0] * mask_rz, sin_out[0] * mask_rz,
        cos_out[1] * mask_rz, sin_out[1] * mask_rz,
        cos_out[2] * mask_l, sin_out[2] * mask_l,
    ))


@partial(jax.custom_jvp, nondiff_argnums=(9, 10, 11, 12, 13))
def _native_projection(packed, cosmui, sinmui, cosmumi, sinmumi, cosnv,
                       sinnv, cosnvn, sinnvn, mpol, ntor, include_edge,
                       asym, threads):
    if _force_ffi is None:
        raise RuntimeError("native VMEX force extension is not available")
    ns, nzeta = int(packed.shape[1]), int(packed.shape[3])
    workers = min(int(threads), ns * int(mpol))
    result, _scratch = ffi.ffi_call(
        "vmex_force_projection",
        (
            jax.ShapeDtypeStruct((6, ns, mpol, ntor + 1), packed.dtype),
            jax.ShapeDtypeStruct((workers, 12, nzeta), packed.dtype),
        ),
        vmap_method="broadcast_all",
    )(
        packed, cosmui, sinmui, cosmumi, sinmumi, cosnv, sinnv, cosnvn, sinnvn,
        mpol=np.int64(mpol), ntor=np.int64(ntor),
        ntheta2=np.int64(cosmui.shape[0]), threads=np.int64(workers),
        include_edge=include_edge, asym=asym,
    )
    return result


@_native_projection.defjvp
def _native_projection_jvp(mpol, ntor, include_edge, asym, threads,
                           primals, tangents):
    value = _native_projection(*primals, mpol, ntor, include_edge, asym, threads)
    _, tangent = jax.jvp(
        lambda *args: _reference(
            *args, mpol=mpol, ntor=ntor,
            include_edge=include_edge, asym=asym,
        ),
        primals,
        tangents,
    )
    return value, tangent


def project_force(
    kernels: dict[str, Any],
    *,
    mpol: int,
    ntor: int,
    trig,
    include_edge: bool = False,
    asym: bool = False,
    backend: str = "jax",
    threads: int = 1,
) -> SpectralForce:
    """Project one symmetry family with ``jax`` or the explicit CPU FFI backend."""
    reference = jnp.asarray(kernels["force_R_even"])
    packed = jnp.stack([
        jnp.zeros_like(reference) if kernels.get(f"{name}_{parity}") is None
        else jnp.asarray(kernels[f"{name}_{parity}"])
        for name in _NAMES for parity in ("even", "odd")
    ])
    args = (
        packed,
        jnp.asarray(trig.cosmui[:trig.ntheta2, :mpol]),
        jnp.asarray(trig.sinmui[:trig.ntheta2, :mpol]),
        jnp.asarray(trig.cosmumi[:trig.ntheta2, :mpol]),
        jnp.asarray(trig.sinmumi[:trig.ntheta2, :mpol]),
        jnp.asarray(trig.cosnv[:, :ntor + 1]),
        jnp.asarray(trig.sinnv[:, :ntor + 1]),
        jnp.asarray(trig.cosnvn[:, :ntor + 1]),
        jnp.asarray(trig.sinnvn[:, :ntor + 1]),
    )
    if threads < 1:
        raise ValueError("threads must be positive")
    if backend == "jax":
        out = _reference(
            *args, mpol=mpol, ntor=ntor, include_edge=include_edge, asym=asym
        )
    elif backend == "native":
        native_args = tuple(jnp.asarray(arg, dtype=reference.dtype) for arg in args)
        out = _native_projection(
            *native_args, mpol, ntor, include_edge, asym, int(threads)
        )
    else:
        raise ValueError("backend must be 'jax' or 'native'")
    second = None if ntor == 0 else out[1]
    fourth = None if ntor == 0 else out[3]
    sixth = None if ntor == 0 else out[5]
    if asym:
        return SpectralForce(
            force_R_sc=out[0], force_R_cs=second,
            force_Z_cc=out[2], force_Z_ss=fourth,
            force_lambda_cc=out[4], force_lambda_ss=sixth,
        )
    return SpectralForce(
        force_R_cc=out[0], force_R_ss=second,
        force_Z_sc=out[2], force_Z_cs=fourth,
        force_lambda_sc=out[4], force_lambda_cs=sixth,
    )
