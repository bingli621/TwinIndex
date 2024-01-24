import matplotlib.pyplot as plt
from Cell import Cell


if __name__ == "__main__":
    # highT cell is monoclinic
    cell = Cell(spgr="P21/c", lattice_params=(9.185, 5.8294, 7.9552, 100.79))
    cell.plot_cell()

    cell.rotate_cell(rot_angles=90, rot_axes=cell.a_vec)
    cell.plot_cell()

    cell.rotate_cell(rot_angles=(90, 90), rot_axes=(cell.a_vec, cell.b_vec))
    cell.plot_cell()

    cell.origin = -cell.c_vec
    cell.plot_cell()

    cell2 = Cell(spgr="P-1", lattice_params=(9.185, 5.8294, 7.9552, 100, 100.79, 100))
    cell2.plot_cell()

    plt.show()
