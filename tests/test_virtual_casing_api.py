"""Public naming contract for prescribed-interface virtual casing."""

import vmex as vj
from vmex.core import freeboundary_diff, virtual_casing


def test_prescribed_interface_has_clear_public_name_and_compatibility_base():
    assert vj.PlasmaVacuumInterface is virtual_casing.PlasmaVacuumInterface
    assert freeboundary_diff.FreeBoundaryDiffProblem is virtual_casing.PlasmaVacuumInterface
    assert vj.surface_field_data_from_state is virtual_casing.surface_field_data_from_state
    assert vj.surface_field_data_from_wout is virtual_casing.surface_field_data_from_wout
