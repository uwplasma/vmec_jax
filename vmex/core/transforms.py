"""VMEC spectral transforms: Fourier <-> real space, force projections.

VMEC2000 counterparts
---------------------
- ``Sources/General/totzsp_mod.f`` (``totzsps``/``totzspa``) — synthesis of the
  real-space geometry fields R, Z, lambda and their poloidal/toroidal
  derivatives from spectral coefficients: :func:`fourier_to_real`.
- ``Sources/General/tomnsp_mod.f`` (``tomnsps``/``tomnspa``) — projection of the
  real-space MHD force kernels back onto the Fourier basis:
  :func:`tomnsps` / :func:`tomnspa`.
- ``Sources/General/symforce.f`` — symmetric/antisymmetric decomposition of the
  force kernels for ``lasym=True`` runs: :func:`symforce_split`.
- ``Sources/Initialization_Cleanup/profil3d.f`` (``scalxc``) — the odd-m
  ``1/sqrt(s)`` internal scaling: :func:`odd_m_sqrt_s_scaling`.

Structure
---------
Production geometry synthesis can use a batched toroidal FFT followed by a
poloidal contraction (with a dense-DFT fallback for undersampled grids).
The solver selects it on measured winning hardware; the public dense synthesis
is retained for x86 CPUs and high-column implicit AD. Force projection uses the
reverse two-stage order. All paths are pure ``jax.numpy``, have no host
round-trips, and are jit-friendly.

Naming: variables use physical names with the VMEC2000 Fortran name recorded
in docstrings/comments (e.g. ``force_R`` for ``armn``, ``force_R_cc`` for
``frcc``).

The numerics are ported verbatim from the parity-proven legacy kernels
``vmex/kernels/tomnsp.py`` (``tomnsps_rzl``/``tomnspa_rzl``) and
``vmex/kernels/realspace.py`` (``vmec_realspace_synthesis*`` /
``vmec_realspace_analysis``); equivalence is enforced in
``tests/test_fourier_transforms_ab.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .fourier import ModeTable, TrigTables

__all__ = [
    "register_pytree_dataclass",
    "SpectralForce",
    "fourier_to_real",
    "real_to_fourier",
    "tomnsps",
    "tomnspa",
    "symforce_split",
    "odd_m_sqrt_s_scaling",
    "physical_to_internal_scale",
]

Array = Any

_DERIVATIVES = ("value", "dtheta", "dzeta")


def register_pytree_dataclass(cls, *, meta: tuple[str, ...] = (),
                              drop: tuple[str, ...] = ()):
    """Register a result dataclass as a JAX pytree (``meta`` fields static).

    The one shared registration helper of all core result containers (R26a;
    formerly copied in fields/forces/geometry/residuals/setup/solver/implicit).

    ``drop`` fields (which must have defaults) are excluded from the pytree
    entirely: they do not appear as leaves *or* in the treedef, and are reset
    to their default on unflatten.  Use for host-side convenience attributes
    (e.g. :attr:`~vmex.core.implicit.ImplicitSolution.runtime`) that must
    not change an established pytree structure.
    """
    names = [f.name for f in dataclass_fields(cls)
             if f.name not in meta and f.name not in drop]
    try:
        return jax.tree_util.register_dataclass(
            cls, data_fields=names, meta_fields=list(meta), drop_fields=list(drop)
        )
    except ValueError as exc:
        # JAX 0.6.2 validates ``drop_fields`` with
        # ``init_fields.difference_update(*drop_fields)``.  A normal
        # ``["runtime"]`` is consequently expanded into characters and the
        # field is reported missing.  Newer JAX releases accept the public
        # sequence-of-names contract above.  Retry only that diagnosed old-JAX
        # failure with singleton iterables, which gives 0.6.2 the intended set
        # subtraction without changing flatten/unflatten semantics.
        if not drop or "Missing fields" not in str(exc):
            raise
        legacy_drop_fields: Any = tuple((name,) for name in drop)
        return jax.tree_util.register_dataclass(
            cls,
            data_fields=names,
            meta_fields=list(meta),
            drop_fields=legacy_drop_fields,
        )


def _einsum(expr: str, *operands: Array) -> Array:
    """Einsum with HIGHEST precision (parity with the legacy kernels)."""
    return jnp.einsum(expr, *operands, precision=lax.Precision.HIGHEST)


# ---------------------------------------------------------------------------
# Coefficient scalings
# ---------------------------------------------------------------------------


def physical_to_internal_scale(modes: ModeTable, trig: TrigTables) -> np.ndarray:
    """Per-mode factor converting physical (wout) to VMEC-internal coefficients.

    VMEC2000: ``fixaray.f`` — internal spectral coefficients are stored divided
    by ``mscale(m) * nscale(|n|)`` because the trig tables carry those factors.
    Returns a static ``(mnmax,)`` array ``1 / (mscale[m] * nscale[|n|])``.
    """
    m_idx = np.asarray(modes.m, dtype=np.int64)
    n_abs = np.abs(np.asarray(modes.n, dtype=np.int64))
    return 1.0 / (np.asarray(trig.mscale)[m_idx] * np.asarray(trig.nscale)[n_abs])


def odd_m_sqrt_s_scaling(s: Array, mpol: int) -> Array:
    """VMEC odd-m ``1/sqrt(s)`` internal scaling factors, shape ``(ns, mpol)``.

    VMEC2000: ``profil3d.f`` (``scalxc``).  Odd-m Fourier coefficients are
    evolved in a ``1/sqrt(s)`` representation:

        ``scalxc(js, m odd) = 1 / max(sqrt(s_js), sqrt(s_2))``

    with ``scalxc = 1`` for even m, ``sqrt(s_ns)`` clamped to exactly 1 at the
    edge, and the axis clipped to the first interior half-mesh value.  The same
    factors multiply the force coefficients after ``tomnsps`` (``funct3d.f``:
    ``gc = gc * scalxc``), making them part of the definition of the reported
    residuals ``fsqr/fsqz/fsql``.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    mpol = int(mpol)
    if ns == 0 or mpol <= 0:
        return jnp.zeros((ns, max(mpol, 0)), dtype=s.dtype)
    sqrt_s = jnp.sqrt(jnp.maximum(s, 0.0))
    # profil3d.f sets sqrts(ns) = 1 exactly to avoid edge roundoff.
    sqrt_s = sqrt_s.at[-1].set(jnp.asarray(1.0, dtype=sqrt_s.dtype))
    sqrt_s_first = sqrt_s[1] if ns >= 2 else jnp.asarray(1.0, dtype=sqrt_s.dtype)
    inv_sqrt_s = 1.0 / jnp.maximum(sqrt_s, sqrt_s_first)
    m_is_odd = (jnp.arange(mpol, dtype=jnp.int32) % 2) == 1
    ones = jnp.ones((ns, mpol), dtype=sqrt_s.dtype)
    return jnp.where(m_is_odd[None, :], inv_sqrt_s[:, None], ones)


def _default_radial_grid(ns: int, dtype: Any) -> Array:
    """Uniform normalized-toroidal-flux grid s in [0, 1] (VMEC full mesh)."""
    if ns < 2:
        return jnp.asarray([0.0], dtype=dtype)
    return jnp.linspace(0.0, 1.0, ns, dtype=dtype)


# ---------------------------------------------------------------------------
# Phase tables (mode-stacked trig products) for synthesis/analysis
# ---------------------------------------------------------------------------


def _phase_pair(modes: ModeTable, trig: TrigTables, derivative: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos_phase, sin_phase)`` tables, each ``(mnmax, ntheta3, nzeta)``.

    ``cos_phase_k = mscale*nscale * cos(m_k*theta - n_k*zeta)`` (and its
    theta/physical-zeta derivatives), assembled from the ``fixaray`` tables via
    the angle-addition identities with the sign of ``n`` handled explicitly
    (tables only store ``n >= 0`` columns).  Built with NumPy: these are static
    trace-time constants.
    """
    m_idx = np.asarray(modes.m, dtype=np.int64)
    n_idx = np.asarray(modes.n, dtype=np.int64)
    n_abs = np.abs(n_idx)
    sign_n = np.where(n_idx < 0, -1.0, 1.0)[:, None, None]

    def theta_cols(table: np.ndarray) -> np.ndarray:
        return np.asarray(table)[:, m_idx].T[:, :, None]  # (mnmax, ntheta3, 1)

    def zeta_cols(table: np.ndarray) -> np.ndarray:
        return np.asarray(table)[:, n_abs].T[:, None, :]  # (mnmax, 1, nzeta)

    if derivative == "value":
        cos_mu, sin_mu = theta_cols(trig.cosmu), theta_cols(trig.sinmu)
        cos_nv, sin_nv = zeta_cols(trig.cosnv), zeta_cols(trig.sinnv)
        cos_phase = cos_mu * cos_nv + sign_n * sin_mu * sin_nv
        sin_phase = sin_mu * cos_nv - sign_n * cos_mu * sin_nv
    elif derivative == "dtheta":
        cos_mum, sin_mum = theta_cols(trig.cosmum), theta_cols(trig.sinmum)
        cos_nv, sin_nv = zeta_cols(trig.cosnv), zeta_cols(trig.sinnv)
        cos_phase = sin_mum * cos_nv + sign_n * cos_mum * sin_nv
        sin_phase = cos_mum * cos_nv - sign_n * sin_mum * sin_nv
    elif derivative == "dzeta":
        cos_mu, sin_mu = theta_cols(trig.cosmu), theta_cols(trig.sinmu)
        cos_nvn, sin_nvn = zeta_cols(trig.cosnvn), zeta_cols(trig.sinnvn)
        cos_phase = cos_mu * sin_nvn + sign_n * sin_mu * cos_nvn
        sin_phase = sin_mu * sin_nvn - sign_n * cos_mu * cos_nvn
    else:
        raise ValueError(f"Unknown derivative {derivative!r}; expected one of {_DERIVATIVES}")
    return cos_phase, sin_phase


def _packed_toroidal_coefficients(
    coefficient_cos: Array, coefficient_sin: Array, modes: ModeTable
) -> tuple[Array, Array, Array, Array]:
    """Repack signed helical modes into separable cos/sin theta-zeta blocks."""
    mpol = int(np.max(np.asarray(modes.m))) + 1
    ntor = int(np.max(np.abs(np.asarray(modes.n)))) if modes.mnmax else 0
    split = ntor + 1
    cos0 = coefficient_cos[..., :split][..., None, :]
    sin0 = coefficient_sin[..., :split][..., None, :]
    shape = (*coefficient_cos.shape[:-1], mpol - 1, 2 * ntor + 1)
    cos_rest = coefficient_cos[..., split:].reshape(shape)
    sin_rest = coefficient_sin[..., split:].reshape(shape)

    cos_neg = cos_rest[..., :ntor][..., ::-1]
    sin_neg = sin_rest[..., :ntor][..., ::-1]
    cos_pos = cos_rest[..., ntor + 1 :]
    sin_pos = sin_rest[..., ntor + 1 :]
    cos_zero = cos_rest[..., ntor : ntor + 1]
    sin_zero = sin_rest[..., ntor : ntor + 1]

    theta_cos_cos = jnp.concatenate(
        [cos_zero, cos_neg + cos_pos], axis=-1
    )
    theta_cos_sin = jnp.concatenate(
        [jnp.zeros_like(sin_zero), sin_neg - sin_pos], axis=-1
    )
    theta_sin_cos = jnp.concatenate(
        [sin_zero, sin_neg + sin_pos], axis=-1
    )
    theta_sin_sin = jnp.concatenate(
        [jnp.zeros_like(cos_zero), cos_pos - cos_neg], axis=-1
    )

    zero = jnp.zeros_like(cos0)
    sin0_zeta = jnp.concatenate(
        [jnp.zeros_like(sin0[..., :1]), -sin0[..., 1:]], axis=-1
    )
    return tuple(
        jnp.concatenate([axis, rest], axis=-2)
        for axis, rest in (
            (cos0, theta_cos_cos),
            (sin0_zeta, theta_cos_sin),
            (zero, theta_sin_cos),
            (zero, theta_sin_sin),
        )
    )


def _toroidal_synthesis(
    coefficient_cos: Array, coefficient_sin: Array, trig: TrigTables
) -> Array:
    """Evaluate one real toroidal Fourier series, using an FFT when unaliased."""
    nzeta = int(np.shape(trig.cosnv)[0])
    n_modes = int(coefficient_cos.shape[-1])
    ntor = n_modes - 1
    if ntor <= (nzeta - 1) // 2:
        factors = jnp.full((n_modes,), nzeta / 2, dtype=coefficient_cos.dtype)
        factors = factors.at[0].set(nzeta)
        spectrum = (
            factors * jnp.asarray(trig.nscale[:n_modes])
            * (coefficient_cos - 1j * coefficient_sin)
        )
        spectrum = jnp.pad(
            spectrum,
            [(0, 0)] * (spectrum.ndim - 1)
            + [(0, nzeta // 2 + 1 - n_modes)],
        )
        return jnp.fft.irfft(spectrum, n=nzeta, axis=-1)
    return (
        _einsum("...mn,kn->...mk", coefficient_cos, trig.cosnv[:, :n_modes])
        + _einsum("...mn,kn->...mk", coefficient_sin, trig.sinnv[:, :n_modes])
    )


def _fast_synthesis(
    coefficient_cos: Array,
    coefficient_sin: Array,
    modes: ModeTable,
    trig: TrigTables,
    derivatives: tuple[str, ...],
) -> tuple[Array, ...]:
    """Use separable FFT synthesis without expanding mode-stacked phases."""
    a_cos, a_sin, b_cos, b_sin = _packed_toroidal_coefficients(
        coefficient_cos, coefficient_sin, modes
    )
    a = _toroidal_synthesis(a_cos, a_sin, trig)
    b = _toroidal_synthesis(b_cos, b_sin, trig)
    mpol = int(a.shape[-2])
    theta = {
        "value": (trig.cosmu[:, :mpol], trig.sinmu[:, :mpol]),
        "dtheta": (trig.sinmum[:, :mpol], trig.cosmum[:, :mpol]),
    }
    fields: list[Array] = []
    for derivative in derivatives:
        if derivative == "dzeta":
            frequency = (
                jnp.arange(a_cos.shape[-1], dtype=coefficient_cos.dtype)
                * float(trig.nfp)
            )
            a_d = _toroidal_synthesis(frequency * a_sin, -frequency * a_cos, trig)
            b_d = _toroidal_synthesis(frequency * b_sin, -frequency * b_cos, trig)
            cosmu, sinmu = trig.cosmu[:, :mpol], trig.sinmu[:, :mpol]
        elif derivative in theta:
            a_d, b_d = a, b
            cosmu, sinmu = theta[derivative]
        else:
            raise ValueError(
                f"Unknown derivative {derivative!r}; expected one of {_DERIVATIVES}"
            )
        fields.append(
            _einsum("...mk,im->...ik", a_d, cosmu)
            + _einsum("...mk,im->...ik", b_d, sinmu)
        )
    return tuple(fields)


# ---------------------------------------------------------------------------
# Fourier -> real space (totzsps/totzspa)
# ---------------------------------------------------------------------------


def fourier_to_real(
    coefficient_cos: Array,
    coefficient_sin: Array,
    *,
    modes: ModeTable,
    trig: TrigTables,
    derivatives: Sequence[str] = _DERIVATIVES,
    internal_coeffs: bool = False,
    odd_m_sqrt_s: bool = False,
    s: Array | None = None,
) -> tuple[Array, ...]:
    """Synthesize real-space fields on the VMEC internal angular grid.

    VMEC2000: ``totzsp_mod.f`` — ``totzsps`` synthesizes the stellarator-
    symmetric blocks (``rmncc/rmnss``, ``zmnsc/zmncs``, ``lmnsc/lmncs``) and
    ``totzspa`` the antisymmetric ones (``rmnsc/rmncs``, ``zmncc/zmnss``,
    ``lmncc/lmnss``).  In the signed-(m, n) mode packing used here (one cos and
    one sin coefficient per mode, ``n`` signed) both reduce to a single
    cos/sin synthesis, so this one function covers R, Z and lambda for both
    parities::

        f(s, theta, zeta) = sum_k [ c_cos_k cos(m_k theta - n_k zeta)
                                  + c_sin_k sin(m_k theta - n_k zeta) ]

    Parameters
    ----------
    coefficient_cos, coefficient_sin:
        Coefficient arrays of shape ``(..., ns, mnmax)``.
    derivatives:
        Which fields to return, from ``("value", "dtheta", "dzeta")``.
        ``dzeta`` is the derivative with respect to the *physical* toroidal
        angle (the ``n*nfp`` factor of VMEC's ``xn``), matching ``totzsps``.
    internal_coeffs:
        If True the inputs are VMEC-internal coefficients (already divided by
        ``mscale*nscale``); if False (default) they are physical/wout
        coefficients and the conversion is applied here.
    odd_m_sqrt_s:
        If True apply the odd-m ``1/sqrt(s)`` factors (``profil3d.f`` scalxc)
        to the coefficients before synthesis — VMEC's ``totzsps`` consumes
        internal coefficients that carry this scaling.
    s:
        Radial grid for ``odd_m_sqrt_s`` (defaults to a uniform [0, 1] grid).

    Returns
    -------
    tuple of arrays, one per requested derivative, each of shape
    ``(..., ns, ntheta3, nzeta)``.
    """
    coeff_cos = jnp.asarray(coefficient_cos)
    coeff_sin = jnp.asarray(coefficient_sin)
    if coeff_cos.ndim < 2 or coeff_sin.ndim < 2:
        raise ValueError("Expected coefficient arrays with shape (..., ns, mnmax)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coefficient_cos and coefficient_sin must have the same shape")
    if int(coeff_cos.shape[-1]) != modes.mnmax:
        raise ValueError("Mode count mismatch between coefficients and mode table")

    if not internal_coeffs:
        scale = jnp.asarray(physical_to_internal_scale(modes, trig), dtype=coeff_cos.dtype)
        scale = scale.reshape((1,) * (coeff_cos.ndim - 1) + (modes.mnmax,))
        coeff_cos = coeff_cos * scale
        coeff_sin = coeff_sin * scale
    if odd_m_sqrt_s:
        ns = int(coeff_cos.shape[-2])
        if s is None:
            s = _default_radial_grid(ns, coeff_cos.dtype)
        mpol = int(np.max(np.asarray(modes.m))) + 1
        scalxc = odd_m_sqrt_s_scaling(s, mpol).astype(coeff_cos.dtype)
        scalxc_mn = scalxc[:, np.asarray(modes.m, dtype=np.int64)]
        scalxc_mn = scalxc_mn.reshape((1,) * (coeff_cos.ndim - 2) + scalxc_mn.shape)
        coeff_cos = coeff_cos * scalxc_mn
        coeff_sin = coeff_sin * scalxc_mn

    coeff = jnp.concatenate([coeff_cos, coeff_sin], axis=-1)
    fields = []
    for derivative in derivatives:
        cos_phase, sin_phase = _phase_pair(modes, trig, derivative)
        phase = jnp.asarray(np.concatenate([cos_phase, sin_phase], axis=0))
        fields.append(_einsum("...k,kij->...ij", coeff, phase))
    return tuple(fields)


def _fourier_to_real_fft(
    coefficient_cos: Array,
    coefficient_sin: Array,
    *,
    modes: ModeTable,
    trig: TrigTables,
    derivatives: Sequence[str] = _DERIVATIVES,
    internal_coeffs: bool = False,
    odd_m_sqrt_s: bool = False,
    s: Array | None = None,
) -> tuple[Array, ...]:
    """Internal separable-FFT variant of :func:`fourier_to_real`."""
    coeff_cos = jnp.asarray(coefficient_cos)
    coeff_sin = jnp.asarray(coefficient_sin)
    if coeff_cos.ndim < 2 or coeff_sin.ndim < 2:
        raise ValueError("Expected coefficient arrays with shape (..., ns, mnmax)")
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coefficient_cos and coefficient_sin must have the same shape")
    if int(coeff_cos.shape[-1]) != modes.mnmax:
        raise ValueError("Mode count mismatch between coefficients and mode table")

    if not internal_coeffs:
        scale = jnp.asarray(
            physical_to_internal_scale(modes, trig), dtype=coeff_cos.dtype
        )
        scale = scale.reshape((1,) * (coeff_cos.ndim - 1) + (modes.mnmax,))
        coeff_cos = coeff_cos * scale
        coeff_sin = coeff_sin * scale
    if odd_m_sqrt_s:
        ns = int(coeff_cos.shape[-2])
        if s is None:
            s = _default_radial_grid(ns, coeff_cos.dtype)
        mpol = int(np.max(np.asarray(modes.m))) + 1
        scalxc = odd_m_sqrt_s_scaling(s, mpol).astype(coeff_cos.dtype)
        scalxc_mn = scalxc[:, np.asarray(modes.m, dtype=np.int64)]
        scalxc_mn = scalxc_mn.reshape(
            (1,) * (coeff_cos.ndim - 2) + scalxc_mn.shape
        )
        coeff_cos = coeff_cos * scalxc_mn
        coeff_sin = coeff_sin * scalxc_mn
    return _fast_synthesis(
        coeff_cos, coeff_sin, modes, trig, tuple(derivatives)
    )


def real_to_fourier(
    field: Array,
    *,
    modes: ModeTable,
    trig: TrigTables,
    parity: str = "both",
) -> tuple[Array, Array]:
    """Project a real-space field on the VMEC internal grid onto Fourier modes.

    The grid-quadrature inverse of :func:`fourier_to_real` (physical/wout
    coefficient convention).  VMEC2000 has no single subroutine for this
    (``tomnsps`` projects *forces*, with extra kernels/masks); this is the
    plain analysis using the same ``fixaray`` integration weights:
    ``dnorm``-normalized full-grid sums for ``lasym=True`` and the
    endpoint-half-weighted reduced grid ``[0, pi]`` for ``lasym=False``.

    On the reduced symmetric grid the cos and sin families are each internally
    orthogonal but *not* mutually orthogonal (cross-talk), so a definite
    ``parity`` should be requested for symmetric fields:

    - ``"cos"``: keep cos coefficients, zero the sin block (R-like fields);
    - ``"sin"``: keep sin coefficients, zero the cos block (Z/lambda-like);
    - ``"both"``: keep both (exact only on the full ``lasym`` grid).

    Returns ``(coefficient_cos, coefficient_sin)``, each ``(ns, mnmax)``.
    """
    f = jnp.asarray(field)
    if f.ndim != 3:
        raise ValueError(f"Expected field with shape (ns, ntheta, nzeta), got {f.shape}")
    if int(f.shape[2]) != int(trig.cosnv.shape[0]):
        raise ValueError("Field zeta grid does not match trig tables")

    m_idx = np.asarray(modes.m, dtype=np.int64)
    n_idx = np.asarray(modes.n, dtype=np.int64)

    use_full_theta = int(trig.ntheta3) != int(trig.ntheta2)
    if use_full_theta:
        # lasym=True: uniform weights on the full [0, 2*pi) grid.
        n_theta = int(trig.ntheta3)
        if int(f.shape[1]) < n_theta:
            raise ValueError("Field theta grid is smaller than VMEC ntheta3")
        weights = np.full((n_theta,), float(trig.dnorm3))
    else:
        # lasym=False: reduced [0, pi] grid with endpoint half-weights.
        n_theta = int(trig.ntheta2)
        if int(f.shape[1]) < n_theta:
            raise ValueError("Field theta grid is smaller than VMEC ntheta2")
        weights = np.full((n_theta,), float(trig.dnorm))
        weights[0] *= 0.5
        weights[n_theta - 1] *= 0.5
    f_weighted = f[:, :n_theta, :] * jnp.asarray(weights, dtype=f.dtype)[None, :, None]

    cos_phase, sin_phase = _phase_pair(modes, trig, "value")
    cos_phase = cos_phase[:, :n_theta, :]
    sin_phase = sin_phase[:, :n_theta, :]

    # Strip mscale*nscale to obtain the unscaled helical basis functions
    # cos(m*theta - n*zeta), sin(m*theta - n*zeta).
    mode_norm = (np.asarray(trig.mscale)[m_idx] * np.asarray(trig.nscale)[np.abs(n_idx)])
    cos_unscaled = jnp.asarray(cos_phase / mode_norm[:, None, None])
    sin_unscaled = jnp.asarray(sin_phase / mode_norm[:, None, None])

    inner_cos = _einsum("sij,kij->sk", f_weighted, cos_unscaled)
    inner_sin = _einsum("sij,kij->sk", f_weighted, sin_unscaled)

    # Grid norms of the unscaled basis: 1 for (m, n) = (0, 0), 1/2 otherwise.
    is_00 = (m_idx == 0) & (n_idx == 0)
    basis_norm = jnp.asarray(np.where(is_00, 1.0, 0.5))
    coeff_cos = inner_cos / basis_norm[None, :]
    coeff_sin = inner_sin / basis_norm[None, :]
    # sin(m*theta - n*zeta) vanishes identically for (m, n) = (0, 0).
    coeff_sin = jnp.where(jnp.asarray(is_00)[None, :], 0.0, coeff_sin)

    parity = str(parity).lower()
    if parity == "cos":
        coeff_sin = jnp.zeros_like(coeff_sin)
    elif parity == "sin":
        coeff_cos = jnp.zeros_like(coeff_cos)
    elif parity != "both":
        raise ValueError("parity must be one of {'cos', 'sin', 'both'}")
    return coeff_cos, coeff_sin


# ---------------------------------------------------------------------------
# Real space -> Fourier force projections (tomnsps/tomnspa)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SpectralForce:
    """Fourier-space force coefficients in VMEC's ``(ns, mpol, ntor+1)`` packing.

    VMEC2000 names (``tomnsp_mod.f``): ``force_R_cc = frcc``, ``force_R_ss =
    frss``, ``force_Z_sc = fzsc``, ``force_Z_cs = fzcs``, ``force_lambda_sc =
    flsc``, ``force_lambda_cs = flcs`` (symmetric, from ``tomnsps``);
    ``force_R_sc = frsc``, ``force_R_cs = frcs``, ``force_Z_cc = fzcc``,
    ``force_Z_ss = fzss``, ``force_lambda_cc = flcc``, ``force_lambda_ss =
    flss`` (antisymmetric, from ``tomnspa``).

    Suffix convention: ``cc`` multiplies ``cos(m theta) cos(n zeta)``, ``ss``
    ``sin sin``, ``sc`` ``sin(m theta) cos(n zeta)``, ``cs`` ``cos sin``.
    Blocks not produced by the transform that built the object (or absent for
    2D runs, ``ntor = 0``) are ``None``.
    """

    force_R_cc: Array | None = None
    force_R_ss: Array | None = None
    force_Z_sc: Array | None = None
    force_Z_cs: Array | None = None
    force_lambda_sc: Array | None = None
    force_lambda_cs: Array | None = None

    force_R_sc: Array | None = None
    force_R_cs: Array | None = None
    force_Z_cc: Array | None = None
    force_Z_ss: Array | None = None
    force_lambda_cc: Array | None = None
    force_lambda_ss: Array | None = None


def _radial_masks(ns: int, mpol: int, include_edge: bool, dtype: Any) -> tuple[Array, Array]:
    """VMEC radial evolution masks for R/Z and lambda force blocks.

    VMEC2000: ``vmec_params.f`` (``jmin2``/``jlam``) and the fixed-boundary
    edge handling in ``tomnsp_mod.f``.  With 1-based surface index ``js``:
    R/Z modes evolve from ``js = jmin2(m)`` (1 for m=0, 2 for m>=1) up to the
    edge only when ``include_edge`` (``getfsq`` ``jedge=1`` semantics); lambda
    evolves from ``js = jlam(m) = 2`` at all surfaces.
    """
    js = np.arange(ns) + 1  # 1-based, as in the Fortran
    m = np.arange(mpol)
    jmin2 = np.where(m == 0, 1, 2)[None, :]
    js_max = ns if include_edge else ns - 1
    mask_rz = (js[:, None] >= jmin2) & (js[:, None] <= js_max)
    mask_lambda = js[:, None] >= 2
    np_dtype = np.dtype(jnp.zeros((), dtype=dtype).dtype)
    return (
        jnp.asarray(mask_rz.astype(np_dtype)[:, :, None]),
        jnp.asarray(mask_lambda.astype(np_dtype)[:, :, None]),
    )


def _tomnsp_theta_stage(
    kernels: dict[str, Array | None],
    *,
    mpol: int,
    trig: TrigTables,
) -> tuple[Array, ...]:
    """Shared theta stage of ``tomnsps``/``tomnspa``: build work arrays w1..w12.

    VMEC2000: the ``work1(:, 1..12)`` arrays of ``tomnsp_mod.f``.  Inputs are
    the real-space force kernels on the internal grid, one (even-m, odd-m)
    pair per channel, shapes ``(ns, ntheta3, nzeta)``:

    - ``force_R`` / ``force_Z``        : ``armn`` / ``azmn`` (multiply the basis),
    - ``force_R_du`` / ``force_Z_du``  : ``brmn`` / ``bzmn`` (multiply d(basis)/d theta),
    - ``force_R_dv`` / ``force_Z_dv``  : ``crmn`` / ``czmn`` (multiply d(basis)/d zeta),
    - ``force_lambda_du/dv``           : ``blmn`` / ``clmn``,
    - ``constraint_R`` / ``constraint_Z``: ``arcon`` / ``azcon`` (spectral-
      condensation constraint force, weighted by ``xmpq = m(m-1)``).

    The theta integration is restricted to the reduced interval
    ``theta in [0, pi]`` (``ntheta2`` points) in both transforms, and the
    even/odd-m planes are combined by selecting per poloidal mode parity.
    Returns the twelve parity-selected work arrays, each ``(3-stack-free)``
    shape ``(ns, mpol, nzeta)``.
    """
    n_theta2 = int(trig.ntheta2)

    reference = kernels["force_R_even"]
    if reference is None:
        raise ValueError("force_R_even is required")
    reference = jnp.asarray(reference)[:, :n_theta2, :]

    def prep(name: str) -> Array:
        value = kernels[name]
        if value is None:
            return jnp.zeros_like(reference)
        return jnp.asarray(value)[:, :n_theta2, :]

    def pair(name: str) -> Array:
        return jnp.stack([prep(f"{name}_even"), prep(f"{name}_odd")], axis=0)

    # Channels projected with the plain integration tables (cosmui/sinmui):
    # order matches the legacy kernel: armn, crmn, azmn, czmn, arcon, azcon, clmn.
    stack_plain = jnp.stack(
        [
            pair("force_R"),
            pair("force_R_dv"),
            pair("force_Z"),
            pair("force_Z_dv"),
            pair("constraint_R"),
            pair("constraint_Z"),
            pair("force_lambda_dv"),
        ],
        axis=0,
    )  # (7, 2, ns, ntheta2, nzeta)
    # Channels projected with the theta-derivative tables (cosmumi/sinmumi):
    # brmn, bzmn, blmn.
    stack_dtheta = jnp.stack(
        [pair("force_R_du"), pair("force_Z_du"), pair("force_lambda_du")],
        axis=0,
    )  # (3, 2, ns, ntheta2, nzeta)

    cosmui = jnp.asarray(trig.cosmui[:n_theta2, :mpol])
    sinmui = jnp.asarray(trig.sinmui[:n_theta2, :mpol])
    cosmumi = jnp.asarray(trig.cosmumi[:n_theta2, :mpol])
    sinmumi = jnp.asarray(trig.sinmumi[:n_theta2, :mpol])

    plain_cos = _einsum("cpsik,im->cpsmk", stack_plain, cosmui)
    plain_sin = _einsum("cpsik,im->cpsmk", stack_plain, sinmui)
    dtheta_sin = _einsum("cpsik,im->cpsmk", stack_dtheta, sinmumi)
    dtheta_cos = _einsum("cpsik,im->cpsmk", stack_dtheta, cosmumi)

    (fR_cos, fRdv_cos, fZ_cos, fZdv_cos, conR_cos, conZ_cos, fLdv_cos) = plain_cos
    (fR_sin, fRdv_sin, fZ_sin, fZdv_sin, conR_sin, conZ_sin, fLdv_sin) = plain_sin
    (fRdu_sin, fZdu_sin, fLdu_sin) = dtheta_sin
    (fRdu_cos, fZdu_cos, fLdu_cos) = dtheta_cos

    # Spectral-condensation constraint multiplier xmpq(m, 1) = m*(m-1)
    # (fixaray.f); weights the arcon/azcon channels.
    m_values = np.arange(mpol, dtype=float)
    xmpq1 = jnp.asarray((m_values * (m_values - 1.0))[None, None, :, None])

    # work1 arrays (tomnsp_mod.f numbering), shape (2, ns, mpol, nzeta).
    w1 = fR_cos + fRdu_sin + xmpq1 * conR_cos
    w2 = -fRdv_cos
    w3 = fR_sin + fRdu_cos + xmpq1 * conR_sin
    w4 = -fRdv_sin
    w5 = fZ_cos + fZdu_sin + xmpq1 * conZ_cos
    w6 = -fZdv_cos
    w7 = fZ_sin + fZdu_cos + xmpq1 * conZ_sin
    w8 = -fZdv_sin
    w9 = fLdu_sin
    w10 = -fLdv_cos
    w11 = fLdu_cos
    w12 = -fLdv_sin

    # Select per-m parity: even-m modes read the even plane, odd-m the odd
    # plane (VMEC mparity = mod(m, 2)).
    m_even = jnp.asarray(((np.arange(mpol) % 2) == 0))[None, :, None]

    def select(w: Array) -> Array:
        return jnp.where(m_even, w[0], w[1])

    return tuple(select(w) for w in (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12))


def _tomnsp_zeta_stage(
    work: tuple[Array, ...],
    *,
    ntor: int,
    trig: TrigTables,
    cos_pairs: tuple[tuple[int, int], ...],
    sin_pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array] | None]:
    """Shared zeta stage: contract the work arrays with the zeta trig tables.

    Each ``cos``-type output is ``wA . cosnv + wB . sinnvn`` and each
    ``sin``-type output is ``wC . sinnv + wD . cosnvn`` (``sinnvn/cosnvn``
    carry the ``-n*nfp`` / ``n*nfp`` derivative factors from ``fixaray.f``).
    The ``sin``-type block exists only for 3D runs (``ntor > 0``, VMEC
    ``lthreed``).
    """
    n_cols = ntor + 1
    cosnv = jnp.asarray(trig.cosnv[:, :n_cols])
    sinnv = jnp.asarray(trig.sinnv[:, :n_cols])
    cosnvn = jnp.asarray(trig.cosnvn[:, :n_cols])
    sinnvn = jnp.asarray(trig.sinnvn[:, :n_cols])
    lthreed = ntor > 0

    w_a = jnp.stack([work[a] for a, _ in cos_pairs], axis=0)
    cos_block = _einsum("csmk,kn->csmn", w_a, cosnv)
    if lthreed:
        w_b = jnp.stack([work[b] for _, b in cos_pairs], axis=0)
        cos_block = cos_block + _einsum("csmk,kn->csmn", w_b, sinnvn)
    cos_out = (cos_block[0], cos_block[1], cos_block[2])

    if not lthreed:
        return cos_out, None
    w_c = jnp.stack([work[c] for c, _ in sin_pairs], axis=0)
    w_d = jnp.stack([work[d] for _, d in sin_pairs], axis=0)
    sin_block = _einsum("csmk,kn->csmn", w_c, sinnv) + _einsum("csmk,kn->csmn", w_d, cosnvn)
    return cos_out, (sin_block[0], sin_block[1], sin_block[2])


def _validate_tomnsp_inputs(force_R_even: Array, trig: TrigTables, mpol: int, ntor: int) -> None:
    shape = jnp.asarray(force_R_even).shape
    if len(shape) != 3:
        raise ValueError("Force kernels must have shape (ns, ntheta3, nzeta)")
    if int(shape[1]) != int(trig.ntheta3) or int(shape[2]) != int(trig.cosnv.shape[0]):
        raise ValueError("Force kernel grid does not match trig tables")
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")


def tomnsps(
    *,
    force_R_even: Array,
    force_R_odd: Array,
    force_R_du_even: Array,
    force_R_du_odd: Array,
    force_R_dv_even: Array,
    force_R_dv_odd: Array,
    force_Z_even: Array,
    force_Z_odd: Array,
    force_Z_du_even: Array,
    force_Z_du_odd: Array,
    force_Z_dv_even: Array,
    force_Z_dv_odd: Array,
    force_lambda_du_even: Array | None = None,
    force_lambda_du_odd: Array | None = None,
    force_lambda_dv_even: Array | None = None,
    force_lambda_dv_odd: Array | None = None,
    constraint_R_even: Array | None = None,
    constraint_R_odd: Array | None = None,
    constraint_Z_even: Array | None = None,
    constraint_Z_odd: Array | None = None,
    mpol: int,
    ntor: int,
    trig: TrigTables,
    include_edge: bool = False,
) -> SpectralForce:
    """Project real-space forces onto the stellarator-symmetric Fourier blocks.

    VMEC2000: ``Sources/General/tomnsp_mod.f``, ``tomnsps`` — the real-space ->
    Fourier-space transform of the MHD force kernels for the symmetric blocks
    ``frcc/frss`` (R), ``fzsc/fzcs`` (Z) and ``flsc/flcs`` (lambda).

    Kernel naming (VMEC2000 names in parentheses; ``_even/_odd`` are the
    poloidal-parity planes, the odd one carrying the sqrt(s) representation):

    - ``force_R`` (``armn``), ``force_Z`` (``azmn``): multiply the basis
      function itself;
    - ``force_R_du`` (``brmn``), ``force_Z_du`` (``bzmn``),
      ``force_lambda_du`` (``blmn``): multiply the poloidal derivative of the
      basis;
    - ``force_R_dv`` (``crmn``), ``force_Z_dv`` (``czmn``),
      ``force_lambda_dv`` (``clmn``): multiply the (physical) toroidal
      derivative of the basis;
    - ``constraint_R`` (``arcon``), ``constraint_Z`` (``azcon``): the spectral-
      condensation constraint force, weighted here by ``xmpq = m(m-1)``.

    ``include_edge=True`` keeps the boundary surface in the R/Z masks
    (``getfsq`` ``jedge=1`` semantics); the default matches fixed-boundary
    evolution (edge row zeroed, ``jmin2``/``jlam`` start indices applied).

    Returns a :class:`SpectralForce` with the symmetric blocks set (the
    ``ss/cs`` blocks are ``None`` for 2D runs, ``ntor = 0``).
    """
    _validate_tomnsp_inputs(force_R_even, trig, mpol, ntor)
    kernels = {
        "force_R_even": force_R_even,
        "force_R_odd": force_R_odd,
        "force_R_du_even": force_R_du_even,
        "force_R_du_odd": force_R_du_odd,
        "force_R_dv_even": force_R_dv_even,
        "force_R_dv_odd": force_R_dv_odd,
        "force_Z_even": force_Z_even,
        "force_Z_odd": force_Z_odd,
        "force_Z_du_even": force_Z_du_even,
        "force_Z_du_odd": force_Z_du_odd,
        "force_Z_dv_even": force_Z_dv_even,
        "force_Z_dv_odd": force_Z_dv_odd,
        "force_lambda_du_even": force_lambda_du_even,
        "force_lambda_du_odd": force_lambda_du_odd,
        "force_lambda_dv_even": force_lambda_dv_even,
        "force_lambda_dv_odd": force_lambda_dv_odd,
        "constraint_R_even": constraint_R_even,
        "constraint_R_odd": constraint_R_odd,
        "constraint_Z_even": constraint_Z_even,
        "constraint_Z_odd": constraint_Z_odd,
    }
    work = _tomnsp_theta_stage(kernels, mpol=mpol, trig=trig)
    # tomnsps wiring (tomnsp_mod.f):
    #   frcc = w1.cosnv + w2.sinnvn ; fzsc = w7.cosnv + w8.sinnvn ; flsc = w11/w12
    #   frss = w3.sinnv + w4.cosnvn ; fzcs = w5.sinnv + w6.cosnvn ; flcs = w9/w10
    cos_out, sin_out = _tomnsp_zeta_stage(
        work,
        ntor=ntor,
        trig=trig,
        cos_pairs=((0, 1), (6, 7), (10, 11)),
        sin_pairs=((2, 3), (4, 5), (8, 9)),
    )
    force_R_cc, force_Z_sc, force_lambda_sc = cos_out
    if sin_out is None:
        force_R_ss = force_Z_cs = force_lambda_cs = None
    else:
        force_R_ss, force_Z_cs, force_lambda_cs = sin_out

    ns = int(jnp.asarray(force_R_even).shape[0])
    mask_rz, mask_lambda = _radial_masks(ns, mpol, include_edge, force_R_cc.dtype)
    maybe = lambda x, mask: None if x is None else x * mask  # noqa: E731
    return SpectralForce(
        force_R_cc=force_R_cc * mask_rz,
        force_R_ss=maybe(force_R_ss, mask_rz),
        force_Z_sc=force_Z_sc * mask_rz,
        force_Z_cs=maybe(force_Z_cs, mask_rz),
        force_lambda_sc=force_lambda_sc * mask_lambda,
        force_lambda_cs=maybe(force_lambda_cs, mask_lambda),
    )


def tomnspa(
    *,
    force_R_even: Array,
    force_R_odd: Array,
    force_R_du_even: Array,
    force_R_du_odd: Array,
    force_R_dv_even: Array,
    force_R_dv_odd: Array,
    force_Z_even: Array,
    force_Z_odd: Array,
    force_Z_du_even: Array,
    force_Z_du_odd: Array,
    force_Z_dv_even: Array,
    force_Z_dv_odd: Array,
    force_lambda_du_even: Array | None = None,
    force_lambda_du_odd: Array | None = None,
    force_lambda_dv_even: Array | None = None,
    force_lambda_dv_odd: Array | None = None,
    constraint_R_even: Array | None = None,
    constraint_R_odd: Array | None = None,
    constraint_Z_even: Array | None = None,
    constraint_Z_odd: Array | None = None,
    mpol: int,
    ntor: int,
    trig: TrigTables,
    include_edge: bool = False,
) -> SpectralForce:
    """Project real-space forces onto the antisymmetric Fourier blocks.

    VMEC2000: ``Sources/General/tomnsp_mod.f``, ``tomnspa`` — used for
    ``lasym=True`` runs after :func:`symforce_split` has separated each kernel
    into its symmetric and antisymmetric parts.  Produces the antisymmetric
    blocks ``frsc/frcs`` (R), ``fzcc/fzss`` (Z) and ``flcc/flss`` (lambda).

    Kernel arguments have the same meaning as in :func:`tomnsps` (pass the
    *antisymmetric* parts here).  The theta/zeta stages are identical; only the
    output wiring differs (the roles of the cos- and sin-projected work arrays
    swap relative to ``tomnsps``).

    Returns a :class:`SpectralForce` with the antisymmetric blocks set (the
    ``cs/ss`` blocks are ``None`` for 2D runs, ``ntor = 0``).
    """
    _validate_tomnsp_inputs(force_R_even, trig, mpol, ntor)
    kernels = {
        "force_R_even": force_R_even,
        "force_R_odd": force_R_odd,
        "force_R_du_even": force_R_du_even,
        "force_R_du_odd": force_R_du_odd,
        "force_R_dv_even": force_R_dv_even,
        "force_R_dv_odd": force_R_dv_odd,
        "force_Z_even": force_Z_even,
        "force_Z_odd": force_Z_odd,
        "force_Z_du_even": force_Z_du_even,
        "force_Z_du_odd": force_Z_du_odd,
        "force_Z_dv_even": force_Z_dv_even,
        "force_Z_dv_odd": force_Z_dv_odd,
        "force_lambda_du_even": force_lambda_du_even,
        "force_lambda_du_odd": force_lambda_du_odd,
        "force_lambda_dv_even": force_lambda_dv_even,
        "force_lambda_dv_odd": force_lambda_dv_odd,
        "constraint_R_even": constraint_R_even,
        "constraint_R_odd": constraint_R_odd,
        "constraint_Z_even": constraint_Z_even,
        "constraint_Z_odd": constraint_Z_odd,
    }
    work = _tomnsp_theta_stage(kernels, mpol=mpol, trig=trig)
    # tomnspa wiring (tomnsp_mod.f):
    #   frsc = w3.cosnv + w4.sinnvn ; fzcc = w5.cosnv + w6.sinnvn ; flcc = w9/w10
    #   frcs = w1.sinnv + w2.cosnvn ; fzss = w7.sinnv + w8.cosnvn ; flss = w11/w12
    cos_out, sin_out = _tomnsp_zeta_stage(
        work,
        ntor=ntor,
        trig=trig,
        cos_pairs=((2, 3), (4, 5), (8, 9)),
        sin_pairs=((0, 1), (6, 7), (10, 11)),
    )
    force_R_sc, force_Z_cc, force_lambda_cc = cos_out
    if sin_out is None:
        force_R_cs = force_Z_ss = force_lambda_ss = None
    else:
        force_R_cs, force_Z_ss, force_lambda_ss = sin_out

    ns = int(jnp.asarray(force_R_even).shape[0])
    mask_rz, mask_lambda = _radial_masks(ns, mpol, include_edge, force_R_sc.dtype)
    maybe = lambda x, mask: None if x is None else x * mask  # noqa: E731
    return SpectralForce(
        force_R_sc=force_R_sc * mask_rz,
        force_R_cs=maybe(force_R_cs, mask_rz),
        force_Z_cc=force_Z_cc * mask_rz,
        force_Z_ss=maybe(force_Z_ss, mask_rz),
        force_lambda_cc=force_lambda_cc * mask_lambda,
        force_lambda_ss=maybe(force_lambda_ss, mask_lambda),
    )


# ---------------------------------------------------------------------------
# Symmetric/antisymmetric decomposition (symforce)
# ---------------------------------------------------------------------------


def symforce_split(
    field: Array,
    *,
    trig: TrigTables,
    reflect_even: bool = True,
) -> tuple[Array, Array]:
    """Split a real-space force kernel into symmetric/antisymmetric parts.

    VMEC2000: ``Sources/General/symforce.f``.  For ``lasym=True`` runs, each
    force kernel on the full theta grid is decomposed under the stellarator
    reflection ``(theta, zeta) -> (2*pi - theta, -zeta)`` before the reduced-
    interval integrations of :func:`tomnsps` (symmetric part) and
    :func:`tomnspa` (antisymmetric part).

    ``reflect_even`` selects the dominant symmetry of the kernel (the mapping
    is not uniform across kernels — see ``symforce.f``):

    - ``True``  (VMEC kinds ``ars/bzs/bls/rcs/czs/cls``, i.e. ``armn``,
      ``bzmn``, ``blmn``, ``arcon``, ``czmn``, ``clmn``):
      ``sym = (a + a_reflected)/2``, ``asym = (a - a_reflected)/2``;
    - ``False`` (VMEC kinds ``brs/azs/zcs/crs``, i.e. ``brmn``, ``azmn``,
      ``azcon``, ``crmn``): the roles are reversed.

    Rows beyond ``ntheta2`` are kept unchanged on the symmetric output (VMEC
    only updates ``i <= ntheta2``; the transforms never read those rows) and
    zeroed on the antisymmetric output.

    Returns ``(field_symmetric, field_antisymmetric)`` with the input shape
    ``(ns, ntheta3, nzeta)``.
    """
    a = jnp.asarray(field)
    if a.ndim != 3:
        raise ValueError("Expected field with shape (ns, ntheta3, nzeta)")
    ns, n_theta3, nzeta = a.shape
    n_theta2 = int(trig.ntheta2)
    n_theta1 = int(trig.ntheta1)
    if int(trig.ntheta3) != int(n_theta3):
        raise ValueError("symforce_split: theta size mismatch with trig tables")

    # Reflection maps (0-based): theta_i -> theta_{ntheta1 - i} (fixed point at
    # i = 0) and zeta_k -> zeta_{nzeta - k mod nzeta}.
    i = np.arange(n_theta2)
    i_reflected = np.where(i == 0, 0, n_theta1 - i)
    k_reflected = (nzeta - np.arange(nzeta)) % nzeta

    a_half = a[:, :n_theta2, :]
    a_reflected = a[:, i_reflected, :][:, :, k_reflected]

    if reflect_even:
        sym_half = 0.5 * (a_half + a_reflected)
        asym_half = 0.5 * (a_half - a_reflected)
    else:
        sym_half = 0.5 * (a_half - a_reflected)
        asym_half = 0.5 * (a_half + a_reflected)

    tail = a[:, n_theta2:, :]
    field_symmetric = jnp.concatenate([sym_half, tail], axis=1)
    field_antisymmetric = jnp.concatenate([asym_half, jnp.zeros_like(tail)], axis=1)
    return field_symmetric, field_antisymmetric
