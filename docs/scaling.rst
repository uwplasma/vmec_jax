Dimensional scaling
===================

Ideal-MHD equilibria have a dimensional similarity transform. VMEX exposes it
for physical orbit studies in which the particle energy and Larmor radius are
fixed, such as 3.5 MeV alpha-particle calculations:

.. code-block:: bash

   vmex --scale input.case
   vmex --scale input.case 1.2 0.8
   vmex --scale wout_case.nc

The optional numbers are positive multiplicative ``B_scale R_scale`` factors.
Without them, VMEX targets the ARIES-CS reference magnitudes
``|b0| = 5.7 T`` and ``Aminor_p = 1.7 m``. A positive magnetic factor preserves
the flux direction. Output names gain ``_scaled``.

After scaling, one command solves the deck, writes WOUT and Boozer files, and
creates ordinary and Boozer-coordinate plots:

.. code-block:: bash

   vmex --plot --booz input.case_scaled

Similarity law
--------------

Let :math:`B_s` and :math:`R_s` be the two factors. Normalized flux, rotational
transform, beta, aspect ratio, spectral mode numbers, and profile shapes do not
change. Dimensional quantities transform as follows.

.. list-table::
   :header-rows: 1
   :widths: 45 25

   * - Quantity
     - Factor
   * - Boundary, magnetic axis, major/minor radii
     - :math:`R_s`
   * - Volume and Jacobian Fourier coefficients
     - :math:`R_s^3`
   * - Magnetic field
     - :math:`B_s`
   * - Covariant field and total/coil currents
     - :math:`B_s R_s`
   * - Contravariant field
     - :math:`B_s/R_s`
   * - Toroidal and poloidal flux
     - :math:`B_s R_s^2`
   * - Pressure and :math:`B^2`
     - :math:`B_s^2`
   * - Magnetic/pressure energy
     - :math:`B_s^2 R_s^3`
   * - :math:`\mathbf J\cdot\mathbf B`
     - :math:`B_s^2/R_s`
   * - VMEC WOUT ``DMerc``, ``DShear``, ``DWell``, ``DCurr``, ``DGeod``
     - :math:`B_s^{-2}R_s^{-4}`
   * - ``IonLarmor``
     - :math:`B_s^{-1}`

For inputs, pressure is scaled once through ``PRES_SCALE``. ``AM``,
``AM_AUX_F``, current/iota profiles, and every normalized spline coordinate
remain shape data. ``CURTOR`` scales only in prescribed-current mode.

ARIES-CS targets from an input
------------------------------

A WOUT contains ``b0`` and ``Aminor_p``, so its factors are exact. A fixed
boundary gives ``Aminor_p`` directly by Fourier quadrature, but ``b0`` depends
on the converged internal field. VMEX therefore runs a bounded radial probe at
``ns <= 9`` and ``ns <= 17`` rather than the requested full ladder. The final
probe uses ``ftol = 1e-10``. The command prints both resolutions and the
coarse-to-fine changes; these changes are the declared target uncertainty.

For a free-boundary input the probe also determines the final minor radius.
The scaled input and mgrid sidecar are then written together. Per-ampere
(``mgrid_mode = S``) field tables scale as :math:`R_s^{-1}`; raw tables scale
as :math:`B_s`, while their recorded currents and ``EXTCUR`` scale as
:math:`B_sR_s`. Direct-coil inputs are rejected because their geometry and
currents must be scaled before field tabulation.

Validation contract
-------------------

The defining test is commutation:

1. solve the original input and scale its WOUT;
2. scale the input (and mgrid when present) and reconverge it;
3. compare every physical WOUT scalar, profile, Fourier coefficient, Mercier
   term, and NESTOR potential/surface field.

VMEX runs this check for finite-pressure prescribed-current fixed boundary and
for symmetric and LASYM free-boundary NESTOR cases. The structured functions
:func:`vmex.core.scaling.scale_input`,
:func:`vmex.core.scaling.scale_mgrid`, and
:func:`vmex.core.scaling.scale_wout` use parsed objects; they never edit
namelist text by prefix.
