from __future__ import annotations

import numpy as np

from vmec_jax.vmec_parity import internal_odd_from_physical, internal_odd_from_physical_vmec_m1


def test_step10_internal_odd_axis_rules_copy_vs_zero():
    # 3-point radial grid, representing js=1 (axis), js=2, js=3.
    s = np.asarray([0.0, 0.25, 1.0], dtype=float)
    sh = np.sqrt(s)

    # Construct a physical odd contribution corresponding to an internal odd field
    # that is well-defined away from the axis.
    odd_int_true = np.asarray([0.0, 3.0, 4.0], dtype=float)
    odd_phys = (sh * odd_int_true)[:, None, None]

    out_copy = np.asarray(internal_odd_from_physical(odd_phys, s, axis="copy_js2"))
    out_zero = np.asarray(internal_odd_from_physical(odd_phys, s, axis="zero"))

    # Away from the axis, both rules must recover the internal field.
    assert np.allclose(out_copy[1:], odd_int_true[1:, None, None])
    assert np.allclose(out_zero[1:], odd_int_true[1:, None, None])

    # On the axis, the rules differ.
    assert np.allclose(out_copy[0], out_copy[1])
    assert np.allclose(out_zero[0], 0.0)


def test_step10_internal_odd_vmec_m1_rule_splits_m1_vs_mge2():
    s = np.asarray([0.0, 0.25, 1.0], dtype=float)
    sh = np.sqrt(s)

    # m=1 piece (should be copied to axis)
    odd_m1_int = np.asarray([0.0, 3.0, 4.0], dtype=float)
    odd_m1_phys = (sh * odd_m1_int)[:, None, None]

    # odd m>=3 piece (must be zero on axis)
    odd_rest_int = np.asarray([0.0, 5.0, 6.0], dtype=float)
    odd_rest_phys = (sh * odd_rest_int)[:, None, None]

    out = np.asarray(internal_odd_from_physical_vmec_m1(odd_m1_phys=odd_m1_phys, odd_mge2_phys=odd_rest_phys, s=s))

    # Away from axis, recover the sum of internal fields.
    assert np.allclose(out[1:], (odd_m1_int[1:] + odd_rest_int[1:])[:, None, None])
    # Axis: only the m=1 internal value is extrapolated.
    assert np.allclose(out[0], odd_m1_int[1])

