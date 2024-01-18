import matplotlib.pyplot as plt
from Cell import Cell


if __name__ == "__main__":
    # highT cell is monoclinic
    cell_highT = Cell(spgr="P21/c", lattice_params=(9.185, 5.8294, 7.9552, 100.79))
    # lowT cell is triclinic
    cell_lowT = Cell(
        spgr="P-1",
        lattice_params=(9.1614, 5.8075, 7.9584, 89.78, 101.00, 91.76),
    )

    cell_lowT.make_as_reference(cell_highT)

    cell_lowT.plot_cell()

    # generate Bragg peaks, defalut is d_min=1
    cell_lowT.plot_peaks(projection="H,0,L", del_Q=0.1, d_min=1)
    cell_lowT.plot_peaks(projection="H,K,0", del_Q=0.1)
    cell_lowT.plot_peaks(projection="0,K,L", del_Q=0.1)

    plt.show()
