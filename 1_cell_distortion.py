import matplotlib.pyplot as plt
import sympy as sp
from Cell import Cell


if __name__ == "__main__":
    # highT cell is monoclinic
    cell_hT = Cell(spgr="P21/c", lattice_params=(9.185, 5.8294, 7.9552, 100.79))
    # lowT cell is triclinic
    cell_lT = Cell(
        spgr="P-1",
        # lattice_params=(9.1614, 5.8075, 7.9584, 89.78, 101.00, 91.76),
        lattice_params=(9.1614, 5.8075, 7.9584, 90, 101.00, 91.76),
    )
    theta1 = -cell_lT.symbol_vars[5] / sp.pi * 180 + 90
    cell_lT.rotate_cell(rot_angles=(theta1,), rot_axes=([0, 0, 1],))
    # theta2 = (
    #     sp.acos(sp.cos(cell_lT.symbol_vars[4]) / sp.sin(cell_lT.symbol_vars[5]))
    #     - cell_hT.symbol_vars[4]
    # )/ sp.pi * 180
    # cell_lT.rotate_cell(rot_angles=(theta1, theta2), rot_axes=([0, 0, 1], [0, 1, 0]))

    cell_lT.make_as_reference(cell_hT, SYMBOL=True)

    cell_lT.plot_cell()

    # generate Bragg peaks, defalut is d_min=1
    cell_lT.plot_peaks(projection="H,-1,L", del_Q=0.1, d_min=1)
    cell_lT.plot_peaks(projection="H,K,0", del_Q=0.1)
    cell_lT.plot_peaks(projection="0,K,L", del_Q=0.3)

    plt.show()
