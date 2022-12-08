#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###





class DAB_Specification:
	"""
	Class to store the DAB specification
	"""
	def __init__(self, V1, V1_min, V1_max, V2, V2_min, V2_max, P_min, P_max, P_nom, n, L_s, L_m,
				 fs, fs_min: float = None, fs_max: float = None, L_c: float = None):
		self.V1 = V1
		self.V1_min = V1_min
		self.V1_max = V1_max
		self.V2 = V2
		self.V2_min = V2_min
		self.V2_max = V2_max

		self.P_min = P_min
		self.P_max = P_max
		self.P_nom = P_nom

		self.n = n
		self.fs = fs
		self.fs_min = fs_min
		self.fs_max = fs_max

		self.L_s = L_s
		self.L_m = L_m
		self.L_c = L_c


class MLN:
	"""
	class to store data of one operating point
	"""

	def __init__(self):
		self.mln_phi = None
		self.mln_ib_il0 = None
		self.mln_ib_il1 = None
		self.mln_ib_il2 = None
		self.mln_ib_il3 = None
		self.mln_ob_il0 = None
		self.mln_ob_il1 = None
		self.mln_ob_il2 = None
		self.mln_ob_il3 = None
		self.mln_ib_l_i_rms = None
		self.mln_ob_l_i_rms = None
		self.mln_ib_te_i_rms = None
		self.mln_ob_te_i_rms = None
		self.mln_phi_zero = None
		self.mln_ib_te_s_i_mean = None
		self.mln_ib_te_d_i_mean = None
		self.mln_ob_te_s_i_mean = None
		self.mln_ob_te_d_i_mean = None
		self.mln_ib_te_s_i_rms = None
		self.mln_ib_te_d_i_rms = None
		self.mln_ob_te_s_i_rms = None
		self.mln_ob_te_d_i_rms = None
		self.mln_ib_ts_s_i_off = None
		self.mln_ob_ts_s_i_off = None
		self.mln_ib_ts_s_e_off = None
		self.mln_ob_ts_s_e_off = None
		self.mln_ib_ts_v_on = None
		self.mln_ib_ts_e_on = None
		self.mln_ob_ts_v_on = None
		self.mln_ob_ts_e_on = None
		self.mln_ib_ts_s_p_cond = None
		self.mln_ib_ts_d_p_cond = None
		self.mln_ob_ts_s_p_cond = None
		self.mln_ob_ts_d_p_cond = None
		self.mln_ib_ts_d_i_off = None
		self.mln_ob_ts_d_i_off = None
		self.mln_ib_ts_d_e_rr = None
		self.mln_ob_ts_d_e_rr = None
		# main inductance
		self.mln_ib_lm_i_peak = None
		self.mln_ib_lm_i_rms = None
		# power
		self.mln_ib_ts_s_p_total = None
		self.mln_ib_ts_d_p_total = None
		self.mln_ob_ts_s_p_total = None
		self.mln_ob_ts_d_p_total = None
		self.mln_p_total = None
		# temperatures
		self.ib_t_int = None
		self.ob_t_int = None
		self.mln_ib_ts_s_tj = None
		self.mln_ib_ts_d_tj = None
		self.mln_ob_ts_s_tj = None
		self.mln_ob_ts_d_tj = None
		self.mfn_ib_ts_sd_tj_cond_max = None
		self.mfn_ob_ts_sd_tj_cond_max = None
		self.mfn_ibob_ts_sd_tj_max_nan_matrix = None
		# basics
		self.v1 = None
		self.v2 = None
		self.power_max = None