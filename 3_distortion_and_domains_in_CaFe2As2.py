import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from Cell import Cell
from Domains import Domains


if __name__ == "__main__":
    # highT cell is tetragognal
    a_T, c_T = 3.879, 11.74
    cell_tet = Cell(spgr="I4/mmm", lattice_params=(a_T, c_T))
    # lowT cell is orthorhombic
    # a_O, b_O, c_O = 5.51, 5.45, 11.66
    # cell_ortho = Cell(
    #     spgr="Fmmm",
    #     lattice_params=(a_O, b_O, c_O),
    # )
    # exaggerated lattice parameters
    a_O, b_O, c_O = a_T * np.sqrt(2) * 1.05, a_T * np.sqrt(2) * 0.95, 11.66
    cell_ortho = Cell(
        spgr="Fmmm",
        lattice_params=(a_O, b_O, c_O),
    )

    # theta = np.arctan(a_O / b_O) / np.pi * 180
    theta = sp.atan(cell_ortho.symbol_vars[0] / cell_ortho.symbol_vars[1]) / sp.pi * 180
    # print(theta)

    cell_ortho.rotate_cell(rot_angles=-theta, rot_axes=(0, 0, 1))
    # cell_ortho.origin = -cell_ortho.b_vec / 2
    cell_ortho.make_as_reference(cell_tet)

    domains = Domains(cell_ortho)
    domains.make_twin(rot_angles=2 * theta, rot_axes=cell_ortho.c_vec)
    domains.make_twin(rot_angles=2 * theta - 90, rot_axes=cell_ortho.c_vec)
    domains.make_twin(rot_angles=90, rot_axes=cell_ortho.c_vec)
    domains.plot_domains()

    # generate Bragg peaks, defalut is d_min=1
    domains.plot_peaks(projection="H,K,0", del_Q=0.1, d_min=1)

    plt.show()
