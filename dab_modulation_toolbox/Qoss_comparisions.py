# import numpy as np
# import dab_datasets as ds
# from debug_tools import *
#
#
# def method1(Coss1: np.ndarray, V1: np.ndarray):
#     C_Par1 = 100e-12
#     Coss1 = Coss1 + C_Par1
#     # Calculate required Q for each voltage
#     Q_AB_req_1 = _integrate_Coss(Coss1 * 2, V1)
#     return Q_AB_req_1
#
#
# def method2(Coss1: np.ndarray, V1: np.ndarray):
#     C_Par1 = np.full_like(Coss1, 100e-12)
#     print(np.size(C_Par1))
#     Qpar = _integrate_Coss(C_Par1 * 2, V1)
#     Q_AB_req_1 = _integrate_Coss(Coss1 * 2, V1)
#     return Qpar+Q_AB_req_1
#
#
# def _integrate_Coss(coss: np.ndarray, V: np.ndarray) -> np.ndarray:
#     """
#     Integrate Coss for each voltage from 0 to V_max
#     :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
#     :return: Qoss(Vds) as one row of data and index = Vds.
#     """
#
#     # Integrate from 0 to v
#     def integrate(v):
#         v_interp = np.arange(v + 1)
#         coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
#         return np.trapz(coss_v)
#
#     coss_int = np.vectorize(integrate)
#     # get an qoss vector that has the resolution 1V from 0 to V_max
#     v_vec = np.arange(coss.shape[0])
#     # get an qoss vector that fits the mesh_V scale
#     # v_vec = np.linspace(V_min, V_max, int(V_step))
#     qoss = coss_int(v_vec)
#
#     # Calculate a qoss mesh that is like the V mesh
#     # Each element in V gets its q(v) value
#     def meshing_q(v):
#         return np.interp(v, v_vec, qoss)
#
#     q_meshgrid = np.vectorize(meshing_q)
#     qoss_mesh = q_meshgrid(V)
#
#     return qoss_mesh
#
#
# # ---------- MAIN ----------
# if __name__ == '__main__':
#
#     dab = ds.DAB_Data()
#     dab.V1_nom = 700
#     dab.V1_min = 600
#     dab.V1_max = 800
#     dab.V1_step = 3
#     dab.V2_nom = 235
#     dab.V2_min = 175
#     dab.V2_max = 295
#     dab.V2_step = 25 * 3
#     # dab.V2_step = 4
#     dab.P_min = -2200
#     dab.P_max = 2200
#     dab.P_nom = 2000
#     dab.P_step = 19 * 3
#     # Generate meshes
#     dab.gen_meshes()
#
#     mosfet1 = 'C3M0065100J'
#     csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
#     csv_file = os.path.join(csv_file, 'Coss_files', f'Coss_{mosfet1}.csv')
#     dab.import_Coss(csv_file, mosfet1)
#
#     Q_1 = method1(dab['coss_' + mosfet1],
#                   dab.mesh_V1)
#     Q_2 = method2(dab['coss_' + mosfet1],
#                   dab.mesh_V1)
#
#     # np.set_printoptions(threshold=100000)
#
#     print(np.min(np.logical_and(Q_1, Q_2)), np.max(np.logical_and(Q_1, Q_2)))


import numpy as np
import matplotlib.pyplot as plt

# Given 4D arrays
array1 = np.array([[[[0.00000000e+00]], [[0.00000000e+00]], [[2.65358979e-06]], [[3.40470000e-01]], [[6.15192654e-01]], [[3.14159265e+00]], [[3.14159531e+00]], [[3.48206265e+00]], [[3.75678531e+00]]]])
array2 = np.array([[[[-2.06286456]], [[-2.06286456]], [[-2.06284367]], [[2.93044648]], [[4.79636467]], [[2.06286456]], [[2.06284367]], [[-2.93044648]], [[-4.79636467]]]])

# Flatten the arrays to 1D
array1_flat = array1.flatten()
array2_flat = array2.flatten()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(array1_flat, array2_flat, marker='o')
plt.title('Plot of Given Arrays')
plt.xlabel('Array 1')
plt.ylabel('Array 2')
plt.grid(True)
plt.show()