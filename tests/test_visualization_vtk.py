from __future__ import annotations

import numpy as np


def test_vts_writer_smoke(tmp_path):
    from vmec_jax.visualization import write_vts_structured_grid

    nx, ny = 2, 3
    x = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    y = 2.0 * x
    z = -x
    v = np.stack([x, y, z], axis=-1)

    out = write_vts_structured_grid(
        tmp_path / "surf.vts",
        x=x,
        y=y,
        z=z,
        point_data={"scalar": x, "vec": v},
    )
    text = out.read_text()
    assert '<VTKFile type="StructuredGrid"' in text
    assert 'WholeExtent="0 1 0 2 0 0"' in text
    assert 'Name="scalar"' in text
    assert 'Name="vec"' in text


def test_vtp_polyline_writer_smoke(tmp_path):
    from vmec_jax.visualization import write_vtp_polyline

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    out = write_vtp_polyline(tmp_path / "line.vtp", points=pts, point_data={"s": np.array([0.0, 1.0, 2.0])})
    text = out.read_text()
    assert '<VTKFile type="PolyData"' in text
    assert 'NumberOfPoints="3"' in text
    assert 'Name="connectivity"' in text
    assert 'Name="offsets"' in text
    assert 'Name="s"' in text

