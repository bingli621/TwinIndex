import sympy as sp
import numpy as np
import numbers
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist import Subplot

from cctbx import crystal, sgtbx, miller


class Cell(object):
    """
    Making a cell of given orientation with given lattice parameters
    ----------------------------------- attributes ------------------------------------
    spgr_symbol                 string of space group
    spgr_info
    symmetry                    string, "cubic", "tetragonal", "orthorhombic", "hexagonal"
                                "monoclinic", "triclinic".
    a, b, c                     symbolic variables
    alpha, beta, gamma          symbolic variables
    lattice_params              NUMERICAL variables in angstrom and degrees
                                automatic populate to (a,b,c,alpha,beta,gamma)
    rot_axis                    axis perpendicular to the plane of a_vec and x=(1,0,0)
    a_vec, b_vec, c_vec         real space vectors
    a_star b_star, c_star       reciprocal space vectors
    alpha_star, beta_star, gamma_star         symbolic variables
    r_mat                       rotation matrix, symbolic
    r_mat_N                     rotation matrix, numerical
    origin                      origin of choice when plooting the unit cell
    rot_angle
    rot_axis
    peaks                       list of (h,k,l) of symmetry-allowed peaks
    Cell_ref                    Cell used as a reference
    conv_mat                    Conversion matrix from (hkl) of Cell to the frame of Cell_ref
    peaks_conv                  converted (hkl) in the frame of Cell_ref
    --------------------------------------- methods -----------------------------------
    latt_vecs                   real space lattice vectors in Cartesian coordinates
    reciprocal_latt_vec         reciprocal space lattice vectors
    rotate_cell(rot_angle, rot_axis) rotate the cell about rot_axis by rot_angle (deg)
    plot_cell(c, label)         c for color, lable for legends
    make_as_reference(Cell_ref) Make Cell_ref as the reference cell
    peaks_to_plot(projection, del_Q=0.1)
    plot_peaks(projection, del_Q=0.1)
    ---------------------------------- static methods -----------------------------------
    rot_vec(theta, n)
    find_rotation(vec1, vec2)
    make_unit_cell
    generate_peaks(spgr_symbol,lattice_params,d_min)       d_min=1 by default

    """

    a_s = sp.Symbol("a", positive=True)
    b_s = sp.Symbol("b", positive=True)
    c_s = sp.Symbol("c", positive=True)
    alpha_s = sp.Symbol("alpha", positive=True)
    beta_s = sp.Symbol("beta", positive=True)
    gamma_s = sp.Symbol("gamma", positive=True)

    def __init__(self, spgr=None, lattice_params=None, origin=None):
        self.spgr_symbol = spgr
        spgr = sgtbx.bravais_types.bravais_lattice(symbol=spgr)
        self.spgr_info = spgr.space_group_info.symbol_and_number()
        self.symmetry = spgr.crystal_system
        self.lattice_params = self.set_latt_params(lattice_params)
        self.origin = origin
        self.rot_angles = ()
        self.rot_axes = ()
        self.latt_vecs()
        self.reciprocal_latt_vec()
        self.r_mat = sp.eye(3)
        self.peaks = None
        self.Cell_ref = None
        self.conv_mat = None
        self.conv_mat_N = None
        self.peaks_conv = None

        sp.init_printing(use_unicode=True)
        print("-" * 50)
        print(
            f"Generating a {self.symmetry} unit cell"
            + f" with space group {self.spgr_info}."
        )

    def set_latt_params(self, lattice_params):
        """Set lattice parammeters based on symmetry"""
        match self.symmetry:
            case "Cubic":
                # a = sp.Symbol("a", positive=True)
                self.a = Cell.a_s
                self.b = Cell.a_s
                self.c = Cell.a_s
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi / 2

                if len(lattice_params) == 1:
                    a_N = lattice_params[0]
                    latt_params = (a_N, a_N, a_N, 90, 90, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Tetragonal":
                # a = sp.Symbol("a", positive=True)
                self.a = Cell.a_s
                self.b = Cell.a_s
                # self.c = sp.Symbol("c", positive=True)
                self.c = Cell.c_s
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi / 2

                if len(lattice_params) == 2:
                    a_N, c_N = lattice_params
                    latt_params = (a_N, a_N, c_N, 90, 90, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Orthorhombic":
                # self.a = sp.Symbol("a", positive=True)
                # self.b = sp.Symbol("b", positive=True)
                # self.c = sp.Symbol("c", positive=True)
                self.a = Cell.a_s
                self.b = Cell.b_s
                self.c = Cell.c_s
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi / 2

                if len(lattice_params) == 3:
                    a_N, b_N, c_N = lattice_params
                    latt_params = (a_N, b_N, c_N, 90, 90, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Hexagonal" | "Trigonal":
                # a = sp.Symbol("a", positive=True)
                self.a = Cell.a_s
                self.b = Cell.a_s
                # self.c = sp.Symbol("c", positive=True)
                self.c = Cell.c_s
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi * 2 / 3

                if len(lattice_params) == 2:
                    a_N, c_N = lattice_params
                    latt_params = (a_N, a_N, c_N, 90, 90, 120)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Monoclinic":
                # self.a = sp.Symbol("a", positive=True)
                # self.b = sp.Symbol("b", positive=True)
                # self.c = sp.Symbol("c", positive=True)
                self.a = Cell.a_s
                self.b = Cell.b_s
                self.c = Cell.c_s
                self.alpha = sp.pi / 2
                # self.beta = sp.Symbol("beta", positive=True)
                self.beta = Cell.beta_s
                self.gamma = sp.pi / 2

                if len(lattice_params) == 4:
                    a_N, b_N, c_N, beta_N = lattice_params
                    latt_params = (a_N, b_N, c_N, 90, beta_N, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Triclinic":
                # self.a = sp.Symbol("a", positive=True)
                # self.b = sp.Symbol("b", positive=True)
                # self.c = sp.Symbol("c", positive=True)
                # self.alpha = sp.Symbol("alpha", positive=True)
                # self.beta = sp.Symbol("beta", positive=True)
                # self.gamma = sp.Symbol("gamma", positive=True)
                self.a = Cell.a_s
                self.b = Cell.b_s
                self.c = Cell.c_s
                self.alpha = Cell.alpha_s
                self.beta = Cell.beta_s
                self.gamma = Cell.gamma_s

                if len(lattice_params) == 6:
                    a_N, b_N, c_N, alpha_N, beta_N, gamma_N = lattice_params
                    latt_params = (a_N, b_N, c_N, alpha_N, beta_N, gamma_N)
        return latt_params

    def latt_vecs(self):
        """
        build lattice vectors in Cartesian coordinate.
        a_vec is generated by rotating x=[1,0,0] by rot_angle along rot_axis.
        b_vec is in the xy-plane
        """

        a1 = self.a
        self.a_vec = sp.trigsimp(sp.Matrix([a1, 0, 0]))

        b1 = sp.trigsimp(self.b * sp.cos(self.gamma))
        b2 = sp.trigsimp(self.b * sp.sin(self.gamma))
        self.b_vec = sp.trigsimp(sp.Matrix([b1, b2, 0]))

        c1 = sp.trigsimp(self.c * sp.cos(self.beta))
        c2 = sp.trigsimp(self.b * self.c * sp.cos(self.alpha) / b2 - b1 * c1 / b2)
        c3 = sp.trigsimp(sp.sqrt(self.c**2 - c1**2 - c2**2))
        self.c_vec = sp.trigsimp(sp.Matrix([c1, c2, c3]))

    def reciprocal_latt_vec(self):
        """build reciprocal lattice vectors"""
        vol = self.a_vec.cross(self.b_vec).dot(self.c_vec)

        self.a_star = sp.trigsimp(2 * sp.pi / vol * self.b_vec.cross(self.c_vec))
        self.b_star = sp.trigsimp(2 * sp.pi / vol * self.c_vec.cross(self.a_vec))
        self.c_star = sp.trigsimp(2 * sp.pi / vol * self.a_vec.cross(self.b_vec))

    def rotate_cell(self, rot_angles=(0,), rot_axes=([0, 0, 1],)):
        """A series of rotations about rot_axes by rot_angles degress"""

        subs_dict = {
            self.a: self.lattice_params[0],
            self.b: self.lattice_params[1],
            self.c: self.lattice_params[2],
            self.alpha: self.lattice_params[3] / 180 * np.pi,
            self.beta: self.lattice_params[4] / 180 * np.pi,
            self.gamma: self.lattice_params[5] / 180 * np.pi,
        }

        # make tuples if only a single rotation
        if not ((type(rot_angles) is tuple) and (type(rot_axes) is tuple)):
            rot_angles = (rot_angles,)
            rot_axes = (rot_axes,)
        # check length consistency if more than one rotation
        elif not len(rot_angles) == len(rot_axes):
            print("Number of angles and axes do NOT match.")

        print("Rotating the cell by the following rotation(s):")
        for i, rot_angle in enumerate(rot_angles):
            rot_axis = sp.Matrix(rot_axes[i])

            if isinstance(rot_angle, numbers.Number):
                rot_angle_N = rot_angle
            else:
                rot_angle_N = float(rot_angle.subs(subs_dict))

            print(f"#{i+1}. Rotate {np.round(rot_angle_N,3)} degrees about axis n =")
            sp.pprint(rot_axis.T)

            self.rot_angles = self.rot_angles + (rot_angle,)
            rot_angle = rot_angle / 180 * sp.pi

            rot_axis = sp.Matrix(rot_axis)
            rot_axis = sp.trigsimp(rot_axis / sp.sqrt(rot_axis.dot(rot_axis)))
            self.rot_axes = self.rot_axes + (rot_axis,)

            # update lattice vectors
            r_mat = Cell.rot_vec(rot_angle, rot_axis)
            self.a_vec = sp.trigsimp(r_mat * self.a_vec)
            self.b_vec = sp.trigsimp(r_mat * self.b_vec)
            self.c_vec = sp.trigsimp(r_mat * self.c_vec)
            self.r_mat = sp.trigsimp(self.r_mat * r_mat)
            # update reciprocal lattice vectors
            self.reciprocal_latt_vec()
        return self

    @staticmethod
    def rot_vec(theta, n):
        """
        Rotate theta (in deg) about unit vector n, ccw
        """

        n = sp.Matrix(n)
        ux, uy, uz = sp.trigsimp(n / sp.sqrt(n.dot(n)))
        c = sp.cos(theta)
        s = sp.sin(theta)
        r_mat = sp.Matrix(
            [
                [
                    c + ux**2 * (1 - c),
                    ux * uy * (1 - c) - uz * s,
                    ux * uz * (1 - c) + uy * s,
                ],
                [
                    uy * ux * (1 - c) + uz * s,
                    c + uy**2 * (1 - c),
                    uy * uz * (1 - c) - ux * s,
                ],
                [
                    uz * ux * (1 - c) - uy * s,
                    uz * uy * (1 - c) + ux * s,
                    c + uz**2 * (1 - c),
                ],
            ]
        )
        r_mat = sp.trigsimp(r_mat)

        return r_mat

    @staticmethod
    def find_rotation(vec1, vec2):
        """retrun angle theta and axis n to rotate vec1 to vec2"""
        vec1 = sp.Matrix(vec1)
        vec2 = sp.Matrix(vec2)
        n = vec1.cross(vec2)
        n = sp.trigsimp(n / sp.sqrt(n.dot(n)))
        theta = sp.trigsimp(
            sp.acos(vec1.dot(vec2) / sp.sqrt(vec1.dot(vec1)) / sp.sqrt(vec2.dot(vec2)))
        )
        return theta, n

    @staticmethod
    def make_unit_cell(Cell):
        """Plot unit cell in real space"""
        subs_dict = {
            Cell.a: Cell.lattice_params[0],
            Cell.b: Cell.lattice_params[1],
            Cell.c: Cell.lattice_params[2],
            Cell.alpha: Cell.lattice_params[3] / 180 * np.pi,
            Cell.beta: Cell.lattice_params[4] / 180 * np.pi,
            Cell.gamma: Cell.lattice_params[5] / 180 * np.pi,
        }
        pts = []
        # shfit origin or not
        if Cell.origin:
            origin = sp.matrix2numpy(Cell.origin.subs(subs_dict), dtype=float).ravel()
        else:
            origin = np.array([0, 0, 0])

        vectors = [
            sp.matrix2numpy(Cell.a_vec.subs(subs_dict), dtype=float).ravel(),
            sp.matrix2numpy(Cell.b_vec.subs(subs_dict), dtype=float).ravel(),
            sp.matrix2numpy(Cell.c_vec.subs(subs_dict), dtype=float).ravel(),
        ]
        a_vec = origin + vectors[0]
        b_vec = origin + vectors[1]
        c_vec = origin + vectors[2]

        pts += [origin, a_vec, b_vec, c_vec]
        pts += [origin + vectors[0] + vectors[1]]
        pts += [origin + vectors[0] + vectors[2]]
        pts += [origin + vectors[1] + vectors[2]]
        pts += [origin + vectors[0] + vectors[1] + vectors[2]]

        pts = np.array(pts)

        positions = [origin, a_vec, b_vec, c_vec]
        points = (pts[:, 0], pts[:, 1], pts[:, 2])
        edges = [
            [pts[0], pts[3], pts[5], pts[1]],
            [pts[1], pts[5], pts[7], pts[4]],
            [pts[4], pts[2], pts[6], pts[7]],
            [pts[2], pts[6], pts[3], pts[0]],
            [pts[0], pts[2], pts[4], pts[1]],
            [pts[3], pts[6], pts[7], pts[5]],
        ]

        return points, positions, edges

    def plot_cell(self, c="C0", label="cell", PLOT=True):
        """Plot the 3D unit cell"""
        points, positions, edges = Cell.make_unit_cell(self)

        faces = Poly3DCollection(edges, linewidths=1, edgecolors=c, label=label)
        faces.set_facecolor(c)
        faces.set_alpha(0.1)

        if PLOT:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.add_collection3d(faces)
            # Plot the pts themselves to force the scaling of the axes
            ax.scatter(*points, s=0)
            # label at the end of a, b, c lattice vectors
            ax.text(*positions[0], s="o", color=c, fontsize=20)
            ax.text(*positions[1], s="a", color=c, fontsize=20)
            ax.text(*positions[2], s="b", color=c, fontsize=20)
            ax.text(*positions[3], s="c", color=c, fontsize=20)
            ax.set_xlabel(r"x ($\AA$)")
            ax.set_ylabel(r"y ($\AA$)")
            ax.set_zlabel(r"z ($\AA$)")
            ax.set_aspect("equal")
            if self.Cell_ref:
                _, _, edges_ref = Cell.make_unit_cell(self.Cell_ref)
                faces_ref = Poly3DCollection(
                    edges_ref, linewidths=1, edgecolors="k", label="Ref"
                )
                faces_ref.set_alpha(0.0)
                ax.add_collection3d(faces_ref)

            ax.legend()

        return points, positions, faces

    def make_as_reference(self, Cell_ref, DIFFERENT_CELL=True):
        """Make Cell_ref as the reference cell when plotting Bragg peaks"""

        if DIFFERENT_CELL:
            subs_dict_ref = {
                Cell.a_s: sp.Symbol("a_r", positive=True),
                Cell.b_s: sp.Symbol("b_r", positive=True),
                Cell.c_s: sp.Symbol("c_r", positive=True),
                Cell.alpha_s: sp.Symbol("alpha_r", positive=True),
                Cell.beta_s: sp.Symbol("beta_r", positive=True),
                Cell.gamma_s: sp.Symbol("gamma_r", positive=True),
            }
            Cell_ref.a = Cell_ref.a.subs(subs_dict_ref)
            Cell_ref.b = Cell_ref.b.subs(subs_dict_ref)
            Cell_ref.c = Cell_ref.c.subs(subs_dict_ref)
            Cell_ref.aplha = Cell_ref.alpha.subs(subs_dict_ref)
            Cell_ref.beta = Cell_ref.beta.subs(subs_dict_ref)
            Cell_ref.gamma = Cell_ref.gamma.subs(subs_dict_ref)

        Cell_ref.latt_vecs()
        Cell_ref.reciprocal_latt_vec()

        self.Cell_ref = Cell_ref
        print(
            f"Choosing the {self.Cell_ref.symmetry} cell as the reference cell"
            + f" for the {self.symmetry} cell."
        )

        self.conv_mat, self.conv_mat_N = Cell.hkl_conversion_mat(self, Cell_ref)
        print("The symbolic conversion matrix to the frame of the reference cell is")
        sp.pprint(self.conv_mat)
        print("The numerical conversion matrix to the frame of the reference cell is")
        sp.pprint(np.round(self.conv_mat_N, 3))

    @staticmethod
    def hkl_conversion_mat(Cell, Cell_ref):
        """
        return the conversoin matrix P_mat
        P*(h,k,l) of Cell gives (h',k'l') in terms of Cell_ref
        """
        # sp.matrix2numpy(cell_ref.a_star.subs(subs_dict), dtype=float).ravel(),
        R_mat = Cell.a_star.row_join(Cell.b_star).row_join(Cell.c_star)
        # sp.pprint(R_mat)
        R_ref_mat = Cell_ref.a_star.row_join(Cell_ref.b_star).row_join(Cell_ref.c_star)
        # sp.pprint(R_ref_mat)
        P_mat = sp.simplify(sp.trigsimp((R_ref_mat**-1) * R_mat))

        subs_dict_N = {
            Cell.a: Cell.lattice_params[0],
            Cell.b: Cell.lattice_params[1],
            Cell.c: Cell.lattice_params[2],
            Cell.alpha: Cell.lattice_params[3] / 180 * np.pi,
            Cell.beta: Cell.lattice_params[4] / 180 * np.pi,
            Cell.gamma: Cell.lattice_params[5] / 180 * np.pi,
            Cell_ref.a: Cell_ref.lattice_params[0],
            Cell_ref.b: Cell_ref.lattice_params[1],
            Cell_ref.c: Cell_ref.lattice_params[2],
            Cell_ref.alpha: Cell_ref.lattice_params[3] / 180 * np.pi,
            Cell_ref.beta: Cell_ref.lattice_params[4] / 180 * np.pi,
            Cell_ref.gamma: Cell_ref.lattice_params[5] / 180 * np.pi,
        }
        P_mat_N = sp.matrix2numpy(P_mat.subs(subs_dict_N), dtype=float)

        return P_mat, P_mat_N

    @staticmethod
    def generate_peaks(spgr_symbol, lattice_params, d_min):
        """Generate symmetry allowed Bragg peaks"""
        ms = miller.build_set(
            crystal_symmetry=crystal.symmetry(
                space_group_symbol=spgr_symbol,
                unit_cell=lattice_params,
            ),
            anomalous_flag=True,
            d_min=d_min,
        )
        # self.peaks = list(ms.indices())
        peaks = list(ms.expand_to_p1().indices())
        sym = sgtbx.bravais_types.bravais_lattice(symbol=spgr_symbol)
        print("-" * 50)
        print(
            f"Generateing Bragg peaks for a {sym.crystal_system} cell,"
            + " with a minimal d-spacing of "
            + f"{np.round(d_min,3)} Angstrom."
        )
        # print(self.peaks)
        return peaks

    def peaks_to_plot(self, projection, del_Q=0.1, d_min=1):
        """
        Generate peaks to plot for the given projection of the given Cell
        all (hkl) converted if Cell_ref exists
        """

        x = []
        y = []
        x_up = []
        y_up = []
        x_down = []
        y_down = []

        if self.Cell_ref is None:  # No refernce cell
            if self.peaks is None:
                # Generate peaks with d_min=1
                peaks_all = Cell.generate_peaks(
                    self.spgr_symbol, self.lattice_params, d_min
                )
                self.peaks = peaks_all
            else:
                # Peaks should have been generated mannually by
                # Cell.generate_peaks(spgr_symbol, lattice_params, d_min=1)
                # print("No peaks found!")
                peaks_all = self.peaks

        else:  # reference cell chosen!
            peaks_original = Cell.generate_peaks(
                self.spgr_symbol, self.lattice_params, d_min
            )
            self.peaks = peaks_original
            # perform converstion
            _, P_mat_N = Cell.hkl_conversion_mat(self, self.Cell_ref)

            peaks_all = []
            for peak in peaks_original:
                peaks_all.append(tuple(P_mat_N @ peak))

        h, k, l = projection.split(",")

        if h.isalpha():
            if k.isalpha():  # HK0
                l = float(l)

                for peak in peaks_all:
                    if peak[2] == l:
                        x.append(peak[0])
                        y.append(peak[1])
                    elif np.all([peak[2] > l, peak[2] < l + del_Q]):
                        x_up.append(peak[0])
                        y_up.append(peak[1])
                    elif np.all([peak[2] > l - del_Q, peak[2] < l]):
                        x_down.append(peak[0])
                        y_down.append(peak[1])

            else:  # H0L
                k = float(k)
                for peak in peaks_all:
                    if peak[1] == k:
                        x.append(peak[0])
                        y.append(peak[2])
                    elif np.all([peak[1] > k, peak[1] < k + del_Q]):
                        x_up.append(peak[0])
                        y_up.append(peak[2])
                    elif np.all([peak[1] > k - del_Q, peak[1] < k]):
                        x_down.append(peak[0])
                        y_down.append(peak[2])

        else:  # (0KL)
            h = float(h)
            for peak in peaks_all:
                if peak[0] == h:
                    x.append(peak[1])
                    y.append(peak[2])
                elif np.all([peak[0] > h, peak[0] < h + del_Q]):
                    x_up.append(peak[1])
                    y_up.append(peak[2])
                elif np.all([peak[0] > h - del_Q, peak[0] < h]):
                    x_down.append(peak[1])
                    y_down.append(peak[2])

        return x, y, x_up, y_up, x_down, y_down

    @staticmethod
    def axes_setup(cell_ref, projection, del_Q):
        """setup the non-orthogonal axes"""

        subs_dict = {
            cell_ref.a: cell_ref.lattice_params[0],
            cell_ref.b: cell_ref.lattice_params[1],
            cell_ref.c: cell_ref.lattice_params[2],
            cell_ref.alpha: cell_ref.lattice_params[3] / 180 * np.pi,
            cell_ref.beta: cell_ref.lattice_params[4] / 180 * np.pi,
            cell_ref.gamma: cell_ref.lattice_params[5] / 180 * np.pi,
        }

        a_star, b_star, c_star = [
            sp.matrix2numpy(cell_ref.a_star.subs(subs_dict), dtype=float).ravel(),
            sp.matrix2numpy(cell_ref.b_star.subs(subs_dict), dtype=float).ravel(),
            sp.matrix2numpy(cell_ref.c_star.subs(subs_dict), dtype=float).ravel(),
        ]

        h, k, l = projection.split(",")

        if h.isalpha():
            if k.isalpha():  # HK0
                l = float(l)
                x_unit, y_unit = np.linalg.norm(a_star), np.linalg.norm(b_star)
                angle = np.arccos(np.dot(a_star, b_star) / x_unit / y_unit).real

                title = "(H, K, " + f" L={np.round(l, 3)}±{np.round(del_Q, 3)})"
                xlab = "H (r.l.u.)"
                ylab = "K (r.l.u.)"

            else:  # H0L
                k = float(k)
                x_unit, y_unit = np.linalg.norm(a_star), np.linalg.norm(c_star)
                angle = np.arccos(np.dot(a_star, c_star) / x_unit / y_unit).real

                title = "(H, " + f"K={np.round(k, 3)}±{np.round(del_Q, 3)}, " + "L)"
                xlab = "H (r.l.u.)"
                ylab = "L (r.l.u.)"

        else:  # (0KL)
            h = float(h)
            x_unit, y_unit = np.linalg.norm(b_star), np.linalg.norm(c_star)
            angle = np.arccos(np.dot(b_star, c_star) / x_unit / y_unit).real

            title = f" (H={np.round(h, 3)}±{np.round(del_Q, 3)}" + ", K, L)"
            xlab = "K (r.l.u.)"
            ylab = "L (r.l.u.)"

        return (x_unit, y_unit, angle, title, xlab, ylab)

    def plot_peaks(self, projection="H,K,0", del_Q=0.1, d_min=1, c="C0"):
        """in the reference frame of Cell_ref"""

        def tr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return x + y / np.tan(recip_angle), y

        def inv_tr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return x - y / np.tan(recip_angle), y

        if self.Cell_ref:  # reference cell chosen!
            plot_params = Cell.axes_setup(self.Cell_ref, projection, del_Q)
            (x_ref, y_ref, _, _, _, _) = self.Cell_ref.peaks_to_plot(
                projection, del_Q, d_min
            )
            (
                x_conv,
                y_conv,
                x_conv_up,
                y_conv_up,
                x_conv_down,
                y_conv_down,
            ) = self.peaks_to_plot(projection, del_Q, d_min)

        else:  # No reference cell
            plot_params = Cell.axes_setup(self, projection, del_Q)
            x, y, x_up, y_up, x_down, y_down = self.peaks_to_plot(
                projection, del_Q, d_min
            )

        x_unit, y_unit, recip_angle, title, xlab, ylab = plot_params

        fig = plt.figure()
        grid_helper = GridHelperCurveLinear(
            (tr, inv_tr),
            grid_locator1=MaxNLocator(integer=True, steps=[1]),
            grid_locator2=MaxNLocator(integer=True, steps=[1]),
        )
        ax_askew = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)
        fig.add_subplot(ax_askew)

        marker_alpha = 0.6
        if self.Cell_ref:  # reference cell chosen!
            s = ax_askew.scatter(
                *tr(x_conv, y_conv), marker="o", label="cell", c=c, alpha=marker_alpha
            )
            s_up = ax_askew.scatter(
                *tr(x_conv_up, y_conv_up),
                marker="^",
                label="cell",
                c=c,
                alpha=marker_alpha,
            )
            s_down = ax_askew.scatter(
                *tr(x_conv_down, y_conv_down),
                marker="v",
                label="cell",
                c=c,
                alpha=marker_alpha,
            )
            s_ref = ax_askew.scatter(
                *tr(x_ref, y_ref), marker="$\u25EF$", c="k", label="ref", alpha=1
            )
            h, l = ax_askew.get_legend_handles_labels()
            ax_askew.legend(
                handles=[(h[0], h[1], h[2]), (h[3])],
                labels=[(l[0]), (l[3])],
                handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)},
                loc=1,
            )

        else:  # No reference cell
            s = ax_askew.scatter(
                *tr(x, y), marker="o", label="cell", c=c, alpha=marker_alpha
            )
            s_up = ax_askew.scatter(
                *tr(x_up, y_up), marker="^", label="cell", c=c, alpha=marker_alpha
            )
            s_down = ax_askew.scatter(
                *tr(x_down, y_down), marker="v", label="cell", c=c, alpha=marker_alpha
            )

            h, l = ax_askew.get_legend_handles_labels()
            ax_askew.legend(
                handles=[
                    (h[0], h[1], h[2]),
                ],
                labels=[
                    (l[0]),
                ],
                handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)},
                loc=1,
            )

        ax_askew.set_aspect(y_unit / x_unit)
        ax_askew.set_title(title)
        ax_askew.set_xlabel(xlab)
        ax_askew.set_ylabel(ylab)

        ax_askew.grid(alpha=0.6)
        ax_askew.set_xlim(-6, 6)
        ax_askew.set_ylim(-6, 6)

        plt.tight_layout()


# ----------------- testing block -----------------
if __name__ == "__main__":
    cell = Cell(spgr="P21/c", lattice_params=(9.185, 5.8294, 7.9552, 100.79))
    cell.plot_cell()
    cell.plot_peaks(projection="H,0,L", del_Q=0.2, d_min=0.5)
    cell.plot_peaks(projection="H,K,0")
    cell.plot_peaks(projection="0,K,L")

    plt.show()
