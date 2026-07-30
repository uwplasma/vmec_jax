wout file reference
===================

:mod:`vmex.core.wout` implements a VMEC2000-compatible netCDF schema with the
names, dimensions, dtypes, and unit conventions required by simsopt,
booz_xform, and other VMEC-ecosystem tools.  A declared variable may still be
fill-valued where its producer is not implemented; those cases are disclosed
below and in :doc:`vmec2000_compatibility`. Use
:func:`vmex.core.wout.read_wout` / :func:`~vmex.core.wout.write_wout`
for IO and :func:`~vmex.core.wout.wout_from_state` to build the dataset
from a converged solver state.

Unit conventions (applied on write, as in ``wrout.f``):

- ``presf, pres, mass, jcuru, jcurv, ctor`` are divided by :math:`\mu_0`;
- ``phipf, chipf`` are multiplied by :math:`2\pi\,\mathrm{signgs}`;
- ``q_factor = 1 / iotaf``;
- ``lmns`` is on the half mesh; ``bsubsmns`` on the full mesh.

Scalars
-------

``version_``, ``input_extension``, ``mgrid_file``, ``pcurr_type``,
``pmass_type``, ``piota_type``, ``wb``, ``wp``, ``gamma``, ``rmax_surf``,
``rmin_surf``, ``zmax_surf``, ``nfp``, ``ns``, ``mpol``, ``ntor``, ``mnmax``,
``mnmax_nyq``, ``niter``, ``itfsq``, ``lasym``, ``lrecon``, ``lfreeb``,
``lmove_axis``, ``lrfp``, ``ier_flag``, ``aspect``, ``betatotal``,
``betapol``, ``betator``,
``betaxis``, ``b0``, ``rbtor0``, ``rbtor``, ``signgs``, ``IonLarmor``,
``volavgB``, ``ctor``, ``Aminor_p``, ``Rmajor_p``, ``volume_p``, ``ftolv``,
``fsql``, ``fsqr``, ``fsqz``, ``nextcur``, ``extcur(:)``, ``mgrid_mode``.

Mode arrays and axis
--------------------

``xm``, ``xn``, ``xm_nyq``, ``xn_nyq`` (with ``xn = n * nfp``);
``raxis_cc``, ``zaxis_cs`` (plus ``raxis_cs``, ``zaxis_cc`` when ``lasym``).

Profile inputs
--------------

``am``, ``ac``, ``ai`` and the spline tables ``am_aux_s/f``, ``ac_aux_s/f``,
``ai_aux_s/f``.

Radial (1D) profiles
--------------------

Full mesh: ``iotaf``, ``q_factor``, ``presf``, ``phi``, ``phipf``, ``chi``,
``chipf``, ``jcuru``, ``jcurv``, ``jdotb``, ``bdotb``, ``bdotgradv``,
``DMerc``, ``DShear``, ``DWell``, ``DCurr``, ``DGeod``, ``equif``.

Half mesh: ``iotas``, ``mass``, ``pres``, ``beta_vol``, ``buco``, ``bvco``,
``vp``, ``specw``, ``phips``, ``over_r``.

Convergence history: ``fsqt(:)``, ``wdot(:)``.

Fourier tables (mode x radius)
------------------------------

Full mesh: ``rmnc``, ``zmns``, ``bsubsmns``, ``currumnc``, ``currvmnc``.

Half mesh: ``lmns``, and the Nyquist-resolution tables ``gmnc``, ``bmnc``,
``bsubumnc``, ``bsubvmnc``, ``bsupumnc``, ``bsupvmnc``.

When ``lasym = T``, all asymmetric partners are written (``rmns``, ``zmnc``,
``lmnc``, ``gmns``, ``bmns``, ``bsubumns``, ``bsubvmns``, ``bsubsmnc``,
``currumns``, ``currvmns``, ``bsupumns``, ``bsupvmns``).

Free-boundary extras
--------------------

When ``lfreeb = T``: ``nextcur``, ``extcur``, ``curlabel``, ``mgrid_mode``
carry the coil-group metadata from the mgrid file. ``curlabel`` uses
VMEC2000's 30-character label dimension. Passing ``result.vacuum`` to
:func:`~vmex.core.wout.wout_from_state` as ``vacuum_output`` writes the
NESTOR potential modes (``potsin``/``xmpot``/``xnpot``) and the four
``*_sur`` surface-field tables with VMEC2000's Nyquist normalization. The
CLI does this automatically. LASYM runs additionally write ``potcos`` and
the four sine ``*_sur`` partners.

Parity with VMEC2000
--------------------

WOUT parity against representative VMEC2000 golden runs is asserted
per-variable with
combined relative + absolute tolerances (CompareWOut-style methodology from
reference validation), with a documented looser bound for ``currumnc/currvmnc``.
This is not a claim that fill-valued or untested modes have parity. See
:doc:`performance` and :doc:`vmec2000_compatibility`.
