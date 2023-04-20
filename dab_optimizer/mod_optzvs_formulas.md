Formulas for the Optimal ZVS Modulation
=======================================

This file lists all formulas for the **OptZVS (Optimal ZVS) Modulation** according to the Paper [IEEE][1].

All names are converted in such way that they can be python variables.

[1]: https://ieeexplore.ieee.org/document/6671445 (
J. Everts, F. Krismer, J. Van den Keybus, J. Driesen and J. W. Kolar,
"Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters,"
in IEEE Transactions on Power Electronics, vol. 29, no. 8, pp. 3954-3970, Aug. 2014, doi: 10.1109/TPEL.2013.2292026.
)


## Mode 1-
"Note that the equations for mode 1− are similar to those of mode 1+"
Mode 1- equations where not given but derived where neccecary.


## Definitions

All small letters are time dependent values.
A trailing _ indicates a value transformed to the primary side, e.g. u2_ or iHF2_

d: the primary side referred voltage conversion ratio: d = Udc2 / Udc1 = U2 / U1
n: Transformer winding ratio n = n1 / n2
ws: omega_s = 2 pi fs


## (6,7) Mode boundary conditions

(6) mode 1+ : -tau1 + pi <= phi <= tau2
()  mode 1- :
(7) mode 2  : tau2 - tau1 <= phi <= 0

## (8,9) Bridge currents

i_HF1 = iL + I

iHF1 (t) = iL (t) + iLc1 (t)
iHF2 (t) = iHF2_ (t) * n = (iL (t) − iLc2_ (t)) * n

## (Table II) HF AC-Link currents

For each switching instant tti = {a, b, c, d} (paper: theta = {alpha, beta, gamma, delta})
theta = ws * t

### Mode 1+
a:
iL = (U1 (d (-tau1 + tau2/2 - phi + pi) - tau1/2) / (ws * L)
iLc1 = (-U1 * tau1/2) / (ws * Lc1)
iLC2_ = (U2_ (tau1 - tau2/2 + phi - pi) / (ws * Lc2_)

b:
iL = (U1 (d * tau2/2 + tau1/2 - tau2 + phi) / (ws * L)
iLc1 = (-U1 (tau1/2 - tau2 + phi)) / (ws * Lc1)
iLC2_ = (-U2_ * tau2/2) / (ws * Lc2_)

c:
iL = (U1 (d (-tau2/2 + phi) + tau1/2) / (ws * L)
iLc1 = (U1 * tau1/2) / (ws * Lc1)
iLC2_ = (U2_ (tau2/2 - phi) / (ws * Lc2_)

d:
iL = (U1 (-d * tau2/2 - tau1/2 - phi + pi) / (ws * L)
iLc1 = (U1 (-tau1/2 - phi + pi)) / (ws * Lc1)
iLC2_ = (U2_ * tau2/2) / (ws * Lc2_)

### Mode 2
a:
iL = (U1 (d * tau2/2 - tau1/2) / (ws * L)
iLc1 = (-U1 * tau1/2) / (ws * Lc1)
iLC2_ = (-U2_ * tau2/2) / (ws * Lc2_)

b:
iL = (U1 (d * tau2/2 + tau1/2 - tau2 + phi) / (ws * L)
iLc1 = (U1 (tau1/2 - tau2 + phi)) / (ws * Lc1)
iLC2_ = (-U2_ * tau2/2) / (ws * Lc2_)

c:
iL = (U1 (-d * tau2/2 + tau1/2) / (ws * L)
iLc1 = (U1 * tau1/2) / (ws * Lc1)
iLC2_ = (U2_ * tau2/2) / (ws * Lc2_)

d:
iL = (U1 (-d * tau2/2 + tau1 + phi) / (ws * L)
iLc1 = (U1 (tau1/2 + phi)) / (ws * Lc1)
iLC2_ = (U2_ * tau2/2) / (ws * Lc2_)


## 





