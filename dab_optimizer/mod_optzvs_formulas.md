Formulas for the Optimal ZVS Modulation
=======================================

This file lists all formulas for the **OptZVS (Optimal ZVS) Modulation** according to the Paper [IEEE][1] and PhD Thesis [2].

All names are converted in such way that they can be python variables.

[1]: https://ieeexplore.ieee.org/document/7762886 (
J. Everts, "Closed-Form Solution for Efficient ZVS Modulation of DAB Converters," in IEEE Transactions on Power Electronics, vol. 32, no. 10, pp. 7561-7576, Oct. 2017, doi: 10.1109/TPEL.2016.2633507.
)
[2]: https://kuleuven.limo.libis.be/discovery/fulldisplay?docid=lirias1731206&context=SearchWebhook&vid=32KUL_KUL:Lirias&search_scope=lirias_profile&tab=LIRIAS&adaptor=SearchWebhook&lang=en (
Everts, Jordi / Driesen, Johan 
Modeling and Optimization of Bidirectional Dual Active Bridge AC-DC Converter Topologies (Modellering en optimalisatie van bidirectionele dual active bridge AC-DC convertor topologieën) 
2014-04-11
)

## ATTENTION - Naming Conflict
Modulation names "mode 1", "mode 2" and "mode 5" are not the same in different papers!
Names used here are:
High Power Flow: mode 1+ : -tau1 + pi <= phi <= tau2
High Power Flow: mode 1- : -tau1 <= phi <= tau2 - pi
Low  Power Flow: mode 2  : tau2 - tau1 <= phi <= 0

## Mode 1-
"Note that the equations for mode 1− are similar to those of mode 1+"
Mode 1- equations where not given but derived where neccecary.


## Definitions

All small letters are time dependent values.
A trailing _ indicates a value transformed to the primary side, e.g. v2_ or iHF2_
All secondary side values are transformed to the primary side, so V2 becomes V2_ and so forth.
Formula numbers refer to [2].
Primary side values are named with 1 and secondary side with 2, e.g. I1 is primary side DC input current.

d: the primary side referred voltage conversion ratio: d = Vdc2 / Vdc1 = V2 / V1
V2_ = n * V2
n: Transformer winding ratio n = n1 / n2
ws: omega_s = 2 pi fs
Q_AB_req1: Q_AB_req_p but with changed naming according to side 1 and 2 naming scheme.


## Predefined Terms

e1 = V2_ * Q_AB_req2 * ws

e2 = n * V1 * np.pi * I1

e3 = n * (V2_ * (Lc2_ + Ls) - V1 * Lc2_)

e4 = 2 * n * np.sqrt(Q_AB_req1 * Ls * np.power(ws, 2) * V1 * Lc1 * (Lc1 + Ls))

e5 = Ls * Lc2_ * ws * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))


## Solution for interval I (mode 2)

tau1 = (np.sqrt(2) * (Lc1 * np.sqrt(V2_ * e3 * e5) + e4 * e3 * 1/n)) / (V1 * e3 * (Lc1 + Ls))

tau2 = np.sqrt((2 * e5) / (V2_ * e3))

phi = (tau2 - tau1) / 2 + (I1 * ws * Ls * np.pi) / (tau2 * V2_)


## Solution for interval II (mode 2)

tau1 = (np.sqrt(2) * (e5 + ws * Ls * Lc2_ * e2 * (V2_ / V1 * (Ls / Lc2_ + 1) - 1))) / (np.sqrt(V2_ * e3 * e5))

tau2 = np.sqrt((2 * e5) / (V2_ * e3))

phi = 0


## Solution for interval III (mode 1+)

tau1 = np.pi

tau2 = np.sqrt((2 * e5) / (V2_ * e3))

phi = (- tau1 + tau2 + np.pi) / 2 - np.sqrt((- np.power((tau2 - np.pi), 2) + tau1 * (2 * np.pi - tau1)) / 4 - (I1 * ws * Ls * np.pi) / V2_)


## Negative Power Flow

Calculate everything with I1 = abs(I1_negative) and recalculate phi afterwards with:

phi = - (tau1 + phi - tau2)


## Predefined Frequency Pattern (optional)

fs_pre = fs_min + (V1 - V1_min) * (fs_max - fs_min) / (V1_fs_lim - V1_min)

V1 >= V1_fs_lim : fs = fs_max
V1 <  V1_fs_lim : fs = fs_pre


## Charge needed to commutate a bridge leg

Q_com_req = integrate_0^Vdc(2 * Coss(v)) dv

Q_AB_req = Q_com_req / 2 + 0.05µC
Applying a margin (0.05 μC) for
component variances and circuit imperfections.

-> numpy: np.trapz









# OLD Maybe delete - OptZVS Paper

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





