import matplotlib.pyplot as plt
from Cell import Cell
from Domains import Domains


if __name__ == "__main__":
    # highT cell is monoclinic
    a, b, c = 9.185, 5.8294, 7.9552
    beta = 100.79
    cell = Cell(spgr="P21/c", lattice_params=(a, b, c, beta))
    # cell.plot_cell()

    # generating multiple domains of the given cell by rotations
    domains = Domains(cell)
    domains.make_twin(rot_angles=180, rot_axes=cell.c_vec, origin=cell.b_vec)
    # domains.make_twin(rot_angles=180, rot_axes=cell.a_vec, origin=cell.b_vec)
    domains.plot_domains()

    # generate Bragg peaks, defalut d_min=1
    domains.plot_peaks(projection="H,0,L", del_Q=0.1, d_min=1)
    domains.plot_peaks(projection="H,K,0", del_Q=0.5)
    domains.plot_peaks(projection="0,K,L", del_Q=0.5)

    plt.show()
