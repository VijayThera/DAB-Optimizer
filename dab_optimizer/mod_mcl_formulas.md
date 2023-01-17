Formulas for the Minimum Conduction Loss Modulation
===================================================

This file lists all formulas for the **MCL (Minimum Conduction Loss) Modulation** according to the Paper [IEEE][1].

All names are converted in such way that they can be python variables.

[1]: https://ieeexplore.ieee.org/document/5776689 (
F. Krismer and J. W. Kolar, "Closed Form Solution for Minimum Conduction Loss Modulation of DAB Converters,"
in IEEE Transactions on Power Electronics, vol. 27, no. 1, pp. 174-188, Jan. 2012, doi: 10.1109/TPEL.2011.2157976.
)

## Normalizations

V_ref = any arbitrary voltage
Z_ref = 2 * pi * f_s * L
I_ref = V_ref / Z_ref
P_ref = V_ref^2 / Z_ref

V1n = V1 / V_ref
V2n = n * V2 / V_ref
Pn = P / P_ref
In_L = I_L / I_ref

Here denotes the lower case "n" that these values are normalized.

## (14) P_max of the DAB with SPS/CPM

Pn_max = (pi * V1n * V2n) / 4

This will effectively limit the inductor L to:

L <= (min(V1) * min(n * V2)) / (8 * fs * P_max)

because of:

P_max = (n * V1 * V2) / (8 * fs * L)

## (20) Mapping

if V1n <= V2n:
    Van = V1n
    Vbn = V2n
    Da = D1
    Db = D2

if V1n > V2n:
    Van = V2n
    Vbn = V1n
    Da = D2
    Db = D1

Note that phi is not mapped!

## (22) Maximum power for TCM

Pn_tcmmax = pi / 2 * (Van^2 * (Vbn - Van)) / Vbn

## (21) TCM phi, Da and Db

if Pn <= Pn_tcmmax:

phi = pi * sgn(Pn) * sqrt( (Vbn - Van) / (2 * Van^2 * Vbn) * abs(Pn) / pi )

Da = abs(phi) / pi * Vbn / (Vbn - Van)

Db = abs(phi) / pi * Van / (Vbn - Van)

## (23) OTM Da and Db

Formula is taken *as-is* from the Paper.

e1 = - (2 * Van^2 + Vbn^2) / (Van^2 + Vbn^2)

e2 = (Van^3 * Vbn + abs(Pn) / pi * (Van^2 + Vbn^2)) / (Van^3 * Vbn + Van * Vbn^3)

e3 = 8 * Van^7 * Vbn^5 - 64 abs(Pn)^3 / pi^3 * (Van^2 + Vbn^2)^3 
- abs(Pn) / pi * Van^4 * Vbn^2 * (4 * Van^2 + Vbn^2) * (4 * Van^2 + 13 * Vbn^2)
+ 16 * Pn^2 / pi^2 * Van * (Van^2 + Vbn^2)^2 * (4 * Van^2 * Vbn + Vbn^3)

e4 = 8 * Van^9 * Vbn^3 - 8 * (abs(Pn) / pi)^3 * (8 * Van^2 - Vbn^2) * (Van^2 + Vbn^2)^2
- 12 * abs(Pn) / pi * Van^6 * Vbn^2 * (4 * Van^2 + Vbn^2)
+ 3 * (Pn / pi)^2 * Van^3 * Vbn * (4 * Van^2 + Vbn^2) * (8 * Van^2 + 5 * Vbn^2)
+ (3 * abs(Pn) / pi)^(3/2) * Van * Vbn^2 * sqrt(e3)

e5 = (2 * Van^6 * Vbn^2 + 2 * abs(Pn) / pi * (4 * Van^2 + Vbn^2) * (abs(Pn) / pi * (Van^2 + Vbn^2) - Van^3 * Vbn)) *
(3 * Van * Vbn * (Van^2 + Vbn^2) * e4^(1/3))^(-1)

e6 = (4 * (Van^3 * Vbn^2 + 2 * Van^5) + 4 * abs(Pn) / pi (Van^2 * Vbn + Vbn^3)) / (Van * (Van^2 + Vbn^2)^2)

e7 = e4^(1/3) / (6 * Van^3 * Vbn + 6 * Van * Vbn^3) + e1^2 / 4 - (2 * e2) / 3 + e5

e8 = 1 / 4 ((-e1^3 - e6) / sqrt(e7) + 3 * e1^2 - 8 * e2 - 4 * e7)

The resulting formulas:

Da = 1 / 2

Db = 1 / 4 * (2 * sqrt(e7) - 2 * sqrt(e8) - e1)

## (24) OTM phi

phi = pi * sgn(Pn) * (1 / 2 - sqrt(Da * (1 - Da) + Db * (1 - Db) - 1 / 4 - abs(Pn) / pi * 1 / (Van * Vbn)))

## (25) Pn_optmax calculation or search

Pn_optmax:
if Db(Pn_) = 1/2: 
    Pn_optmax = Pn_
only for Pn_tcmmax < Pn_ <= Pn_max

## (26) SPS/CPM phi, Da and Db

if abs(Pn) > Pn_optmax and abs(Pn) <= Pn_max:

phi = pi * sgn(Pn) * (1 / 2 - sqrt(1 / 4 - abs(Pn) / pi * 1 / (Van * Vbn)))

Da = 1/2

Db = 1/2
