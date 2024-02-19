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
    --------------------------------class attributes ----------------------------------
    new_cell_counter            to counter how many new cells have been generated
    subs_dict                   dictionary of all symbolic variables
    ----------------------------------- attributes ------------------------------------
    symbol_vars                 symbolic variable with subindices
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
    rotate_cell(rot_angles, rot_axes) rotate the cell about rot_axis by rot_angle (deg)
    mirror_cell(plane=(h,k,l))  mirror cell with respect to plane (h,k,l)
    translate_cell              translation of origin
    plot_cell(c, label)         c for color, lable for legends
    make_as_reference(Cell_ref) Make Cell_ref as the reference cell
    peaks_to_plot(projection, del_Q=0.1)
    plot_peaks(projection, del_Q=0.1)
    ---------------------------------- static methods -----------------------------------
    variable_init(i)            Initiate symbolic variables
    rot_vec(theta, n)
    find_rotation(vec1, vec2)
    make_unit_cell
    generate_peaks(spgr_symbol,lattice_params,d_min)       d_min=1 by default

    """

    new_cell_counter = 0
    subs_dict = {}

    def __init__(
        self, spgr=None, lattice_params=None, NEW_CELL=True, keep_symbolic_indices=None
    ):
        """
        Initialize a cell with given space group and lattice parameters.
        NEW_CELL flag should be set as False if a twin domain is being
        created.
        """
        sp.init_printing(use_unicode=True)

        if NEW_CELL:
            self.symbol_vars = Cell.symbolic_variable_init(Cell.new_cell_counter + 1)
            if keep_symbolic_indices:  # keep the symbolic varible as the previous one
                for i in keep_symbolic_indices:
                    self.symbol_vars[i] = Cell.symbolic_variable_init(
                        Cell.new_cell_counter
                    )[i]
            Cell.new_cell_counter = Cell.new_cell_counter + 1

        else:  # a twin domain
            self.symbol_vars = Cell.symbolic_variable_init(Cell.new_cell_counter)

        self.cell_index = Cell.new_cell_counter
        self.spgr_symbol = spgr
        spgr = sgtbx.bravais_types.bravais_lattice(symbol=spgr)
        self.spgr_info = spgr.space_group_info.symbol_and_number()
        self.symmetry = spgr.crystal_system
        self.lattice_params = self.set_latt_params(lattice_params)
        self.origin = None
        self.rot_angles = ()
        self.rot_axes = ()
        self.latt_vecs()
        self.reciprocal_latt_vec()
        self.r_mat = sp.eye(3)
        self.peaks = None
        self.Cell_ref = None
        self.REF_SYMBOLIC = None
        self.conv_mat = None
        self.conv_mat_N = None
        self.peaks_conv = None

        print("-" * 50)
        print(
            f"Generating a {self.symmetry} unit cell (#{self.cell_index})"
            + f" with space group {self.spgr_info}."
        )

        Cell.subs_dict = Cell.subs_dict | Cell.numerical_subs_dict(self)

    @staticmethod
    def symbolic_variable_init(i):
        """Generate symbolic variable with subindex i"""
        a = sp.Symbol(f"a{i}", positive=True)
        b = sp.Symbol(f"b{i}", positive=True)
        c = sp.Symbol(f"c{i}", positive=True)
        alpha = sp.Symbol(f"alpha{i}", positive=True)
        beta = sp.Symbol(f"beta{i}", positive=True)
        gamma = sp.Symbol(f"gamma{i}", positive=True)
        sym_vars = [a, b, c, alpha, beta, gamma]
        return sym_vars

    @staticmethod
    def numerical_subs_dict(Cell):
        """substitution dictionary numerical"""
        subs_dict = {
            Cell.a: Cell.lattice_params[0],
            Cell.b: Cell.lattice_params[1],
            Cell.c: Cell.lattice_params[2],
            Cell.alpha: Cell.lattice_params[3] / 180 * np.pi,
            Cell.beta: Cell.lattice_params[4] / 180 * np.pi,
            Cell.gamma: Cell.lattice_params[5] / 180 * np.pi,
        }
        return subs_dict

    def set_latt_params(self, lattice_params):
        """Set lattice parammeters based on symmetry"""
        match self.symmetry:
            case "Cubic":
                a_s = self.symbol_vars[0]
                self.a = a_s
                self.b = a_s
                self.c = a_s
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi / 2

                if len(lattice_params) == 1:
                    a_N = lattice_params[0]
                    latt_params = (a_N, a_N, a_N, 90, 90, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Tetragonal":
                a_s = self.symbol_vars[0]
                self.a = a_s
                self.b = a_s
                self.c = self.symbol_vars[2]
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi / 2

                if len(lattice_params) == 2:
                    a_N, c_N = lattice_params
                    latt_params = (a_N, a_N, c_N, 90, 90, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Orthorhombic":
                self.a = self.symbol_vars[0]
                self.b = self.symbol_vars[1]
                self.c = self.symbol_vars[2]
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi / 2

                if len(lattice_params) == 3:
                    a_N, b_N, c_N = lattice_params
                    latt_params = (a_N, b_N, c_N, 90, 90, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Hexagonal" | "Trigonal":
                a_s = self.symbol_vars[0]
                self.a = a_s
                self.b = a_s
                self.c = self.symbol_vars[2]
                self.alpha = sp.pi / 2
                self.beta = sp.pi / 2
                self.gamma = sp.pi * 2 / 3

                if len(lattice_params) == 2:
                    a_N, c_N = lattice_params
                    latt_params = (a_N, a_N, c_N, 90, 90, 120)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

            case "Monoclinic":
                if len(lattice_params) == 4:
                    a_N, b_N, c_N, beta_N = lattice_params
                    latt_params = (a_N, b_N, c_N, 90, beta_N, 90)
                elif len(lattice_params) == 6:
                    latt_params = lattice_params

                self.a = self.symbol_vars[0]
                self.b = self.symbol_vars[1]
                self.c = self.symbol_vars[2]
                self.alpha = sp.pi / 2
                if isinstance(latt_params[4], int):
                    self.beta = latt_params[4] / 180 * sp.pi
                else:
                    self.beta = self.symbol_vars[4]
                self.gamma = sp.pi / 2

            case "Triclinic":
                if len(lattice_params) == 6:
                    a_N, b_N, c_N, alpha_N, beta_N, gamma_N = lattice_params
                    latt_params = (a_N, b_N, c_N, alpha_N, beta_N, gamma_N)

                self.a = self.symbol_vars[0]
                self.b = self.symbol_vars[1]
                self.c = self.symbol_vars[2]

                if isinstance(latt_params[3], int):
                    self.alpha = latt_params[3] / 180 * sp.pi
                else:
                    self.alpha = self.symbol_vars[3]
                if isinstance(latt_params[4], int):
                    self.beta = latt_params[4] / 180 * sp.pi
                else:
                    self.beta = self.symbol_vars[4]
                if isinstance(latt_params[5], int):
                    self.gamma = latt_params[5] / 180 * sp.pi
                else:
                    self.gamma = self.symbol_vars[5]

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

        subs_dict = Cell.subs_dict

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

    def translate_cell(self, vec=None):
        """Translate cell origin by vector vec"""
        h, k, l = vec
        self.origin = h * self.a_vec + k * self.b_vec + l * self.c_vec
        if vec:
            print(f"Translating cell #{self.cell_index} by the vector v=")
            sp.pprint(self.origin)
        else:
            print("Translation of the origin not implemented.")

    def mirror_cell(self, plane=(1, 0, 0)):
        """Mirror cell by plane (h,k,l)"""
        h, k, l = plane
        print(f"Mirror cell #{self.cell_index} with respect to plane ({h}, {k}, {l}).")
        # ------------- find norm ------------------
        if h == 0:
            if k == 0:  # (0,0,l)
                norm = self.c_star

            else:
                if l == 0:  # (0,k,0)
                    norm = self.b_star
                else:  # (0,k,l)
                    vec = self.c_vec / l - self.b_vec / k
                    norm = vec.cross(self.a_vec)
        else:
            if k == 0:
                if l == 0:  # (h,0,0)
                    norm = self.a_star
                else:  # (h,0,l)
                    vec = self.c_vec / l - self.a_vec / h
                    norm = vec.cross(self.b_vec)
            else:
                if l == 0:  # (h,k,0)
                    vec = self.b_vec / k - self.a_vec / h
                    norm = vec.cross(self.c_vec)
                else:  # (h,k,l)
                    vec0 = self.b_vec / k - self.a_vec / h
                    vec1 = self.c_vec / l - self.a_vec / h
                    norm = vec0.cross(vec1)

        norm = sp.trigsimp(norm / sp.sqrt(norm.dot(norm)))
        # print(norm)
        # ------------mirror latt_vec-----------
        r_mat = Cell.rot_vec(sp.pi, norm)
        self.a_vec = sp.trigsimp(r_mat * self.a_vec)
        self.b_vec = sp.trigsimp(r_mat * self.b_vec)
        self.c_vec = sp.trigsimp(r_mat * self.c_vec)
        self.r_mat = sp.trigsimp(self.r_mat * r_mat)
        # update reciprocal lattice vectors
        self.reciprocal_latt_vec()

        return self

    @staticmethod
    def make_unit_cell(Cell):
        """Plot unit cell in real space"""
        subs_dict = Cell.subs_dict

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

    def make_as_reference(self, Cell_ref, SYMBOL=False):
        """
        Make Cell_ref as the reference cell when plotting Bragg peaks
        Give symbolic expressions if SYMBOL=True
        """

        Cell_ref.latt_vecs()
        Cell_ref.reciprocal_latt_vec()

        self.Cell_ref = Cell_ref
        self.REF_SYMBOLIC = SYMBOL
        print(
            f"Choosing the {self.Cell_ref.symmetry} cell (#{self.Cell_ref.cell_index}) "
            + f"as the reference cell for the {self.symmetry} cell (#{self.cell_index})."
        )

        if SYMBOL:
            self.conv_mat, self.conv_mat_N = Cell.hkl_conversion_mat(
                self, Cell_ref, SYMBOL
            )
            print(
                "The symbolic conversion matrix to the frame of the reference cell "
                + f"(#{self.Cell_ref.cell_index}) is"
            )
            sp.pprint(self.conv_mat)
        else:
            self.conv_mat_N = Cell.hkl_conversion_mat(self, Cell_ref, SYMBOL)
        print(
            "The numerical conversion matrix to the frame of the reference cell "
            + f"(#{self.Cell_ref.cell_index}) is"
        )
        sp.pprint(np.round(self.conv_mat_N, 3))

    @staticmethod
    def hkl_conversion_mat(Cell, Cell_ref, SYMBOL=False):
        """
        return the conversoin matrix P_mat
        P*(h,k,l) of Cell gives (h',k'l') in terms of Cell_ref
        """
        # sp.matrix2numpy(cell_ref.a_star.subs(subs_dict), dtype=float).ravel(),
        R_mat = Cell.a_star.row_join(Cell.b_star).row_join(Cell.c_star)
        # sp.pprint(R_mat)
        R_ref_mat = Cell_ref.a_star.row_join(Cell_ref.b_star).row_join(Cell_ref.c_star)
        # sp.pprint(R_ref_mat)

        if SYMBOL:
            # mP_mat = sp.simplify(sp.trigsimp((R_ref_mat**-1) * R_mat))
            P_mat = sp.simplify(sp.trigsimp((R_ref_mat**-1) * R_mat))
            # subs_dict_N = Cell.numerical_subs_dict(Cell) | Cell.numerical_subs_dict(Cell_ref)
            P_mat_N = sp.matrix2numpy(P_mat.subs(Cell.subs_dict), dtype=float)

            return P_mat, P_mat_N
        else:
            # R_mat_N = sp.matrix2numpy(R_mat.subs(Cell.subs_dict), dtype=float)
            # R_ref_mat_N = sp.matrix2numpy(R_ref_mat.subs(Cell.subs_dict), dtype=float)
            # P_mat_N =  sp.simplify(sp.trigsimp((R_ref_mat_N**-1)) * R_mat_N)

            P_mat = (R_ref_mat**-1) * R_mat
            P_mat_N = sp.matrix2numpy(P_mat.subs(Cell.subs_dict), dtype=float)

            return P_mat_N

    @staticmethod
    def generate_peaks(Cell, d_min):
        """Generate symmetry allowed Bragg peaks"""
        ms = miller.build_set(
            crystal_symmetry=crystal.symmetry(
                space_group_symbol=Cell.spgr_symbol,
                unit_cell=Cell.lattice_params,
            ),
            anomalous_flag=True,
            d_min=d_min,
        )
        # self.peaks = list(ms.indices())
        peaks = list(ms.expand_to_p1().indices())
        sym = sgtbx.bravais_types.bravais_lattice(symbol=Cell.spgr_symbol)
        print("-" * 50)
        print(
            f"Generating Bragg peaks for the {sym.crystal_system} cell "
            + f"(#{Cell.cell_index}),"
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
                peaks_all = Cell.generate_peaks(self, d_min)
                self.peaks = peaks_all
            else:
                # Peaks should have been generated mannually by
                # Cell.generate_peaks(Cell, d_min=1)
                # print("No peaks found!")
                peaks_all = self.peaks

        else:  # reference cell chosen!
            peaks_original = Cell.generate_peaks(self, d_min)
            self.peaks = peaks_original
            # perform converstion
            P_mat_N = Cell.hkl_conversion_mat(self, self.Cell_ref, SYMBOL=False)

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

        subs_dict = Cell.subs_dict

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
