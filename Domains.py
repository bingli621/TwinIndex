import sympy as sp
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist import Subplot

from Cell import Cell


class Domains(object):
    """
    Making twins of given cells by rotating the cell
    -------------------------------- attributes -----------------------------------
    Cell_original           cell used to generate twins
    Cell_ref                refernece cell for reciprocal space plotting
    domains_list            List of all domains generated
    -------------------------------- methods --------------------------------------
    make_twin_rotation(rot_angles,rot_axes,origin)  generate a twin by a series of rotations
                                                    about given axes, shift origin
    make_twin_mirror(plane=(h,k,l),origin)                                              
    plot_domains
    plot_peaks
    """

    def __init__(self, Cell):
        self.Cell_original = Cell
        self.domains_list = [
            Cell,
        ]
        if Cell.Cell_ref:
            self.Cell_ref = Cell.Cell_ref
        else:
            self.Cell_ref = None

    def make_twin_rotation(self, rot_angles=(), rot_axes=(), origin=None):
        """
        Generate a twin with the cell parameters of Cell_twin.
        The orientation of Cell_twin is achieved by a series of rotations
        prescirbed by rotating abut rot_axes by rot_angles, concecutively.
        """
        cell = self.Cell_original
        cell_twin = Cell(cell.spgr_symbol, cell.lattice_params, NEW_CELL=False)

        if not ((type(rot_angles) is tuple) and (type(rot_axes) is tuple)):
            rot_angles = (rot_angles,)
            rot_axes = (rot_axes,)

        angles = cell.rot_angles + rot_angles
        axes = cell.rot_axes + rot_axes
        if angles:
            cell_twin.rotate_cell(angles, axes)

        if origin:
            cell_twin.translate_cell(origin)

        # choose the reference cell
        if cell.Cell_ref:
            cell_twin.make_as_reference(cell.Cell_ref)
        else:
            cell_twin.make_as_reference(self.domains_list[0])
        self.domains_list.append(cell_twin)
    
    def make_twin_mirror(self, plane=(1,0,0), origin=None):
        """
        Generate a twin with the cell parameters of Cell_twin.
        The orientation of Cell_twin is achieved by mirroring w.r.t. plane (h,k,l),
        shift the origin 
        """
        cell = self.Cell_original
        cell_twin = Cell(cell.spgr_symbol, cell.lattice_params, NEW_CELL=False)

        if cell.rot_angles:
            cell_twin.rotate_cell(cell.rot_angles, cell.rot_axes)
        cell_twin = cell_twin.mirror_cell(plane)
        if origin:
            cell_twin.translate_cell(origin)

        # choose the reference cell
        if cell.Cell_ref:
            cell_twin.make_as_reference(cell.Cell_ref)
        else:
            cell_twin.make_as_reference(self.domains_list[0])
        self.domains_list.append(cell_twin)



    def plot_domains(self):
        """Plot refence cell and all domians"""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i, domain in enumerate(self.domains_list):
            c = f"C{i}"

            points, positions, faces = domain.plot_cell(
                c=c, label=f"domain_{i+1}", PLOT=False
            )

            ax.add_collection3d(faces)
            # Plot the pts themselves to force the scaling of the axes
            ax.scatter(*points, s=0)
            # label at the end of a, b, c lattice vectors
            ax.text(*positions[0] + i * 0.3, s=f"$o_{i+1}$", color=c, fontsize=20)
            ax.text(*positions[1] + i * 0.3, s=f"$a_{i+1}$", color=c, fontsize=20)
            ax.text(*positions[2] + i * 0.3, s=f"$b_{i+1}$", color=c, fontsize=20)
            ax.text(*positions[3] + i * 0.3, s=f"$c_{i+1}$", color=c, fontsize=20)

        ax.set_xlabel(r"x ($\AA$)")
        ax.set_ylabel(r"y ($\AA$)")
        ax.set_zlabel(r"z ($\AA$)")
        ax.set_aspect("equal")
        plt.tight_layout()

        if self.Cell_ref:
            _, _, edges_ref = Cell.make_unit_cell(self.Cell_ref)
            faces_ref = Poly3DCollection(
                edges_ref, linewidths=1, edgecolors="k", label="Ref"
            )
            faces_ref.set_alpha(0.0)
            ax.add_collection3d(faces_ref)

        ax.legend()

    def plot_peaks(self, projection=("H,K,0"), del_Q=0.1, d_min=1, c="C0"):
        """
        plot Bragg peaks in the given projection plane
        in the reference frame of Cell_ref
        """

        def tr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return x + y / np.tan(recip_angle), y

        def inv_tr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return x - y / np.tan(recip_angle), y

        if self.Cell_ref:  # reference cell chosen!
            xs_conv = []
            ys_conv = []
            xs_conv_up = []
            ys_conv_up = []
            xs_conv_down = []
            ys_conv_down = []
            plot_params = Cell.axes_setup(self.Cell_ref, projection, del_Q)
            (x_ref, y_ref, _, _, _, _) = self.Cell_ref.peaks_to_plot(
                projection, del_Q, d_min
            )
            for domain in self.domains_list:
                (
                    x_conv,
                    y_conv,
                    x_conv_up,
                    y_conv_up,
                    x_conv_down,
                    y_conv_down,
                ) = domain.peaks_to_plot(projection, del_Q, d_min)
                xs_conv.append(x_conv)
                ys_conv.append(y_conv)
                xs_conv_up.append(x_conv_up)
                ys_conv_up.append(y_conv_up)
                xs_conv_down.append(x_conv_down)
                ys_conv_down.append(y_conv_down)

        else:  # No reference cell
            xs = []
            ys = []
            xs_up = []
            ys_up = []
            xs_down = []
            ys_down = []
            plot_params = Cell.axes_setup(self.Cell_original, projection, del_Q)

            for i, domain in enumerate(self.domains_list):
                x, y, x_up, y_up, x_down, y_down = domain.peaks_to_plot(
                    projection, del_Q, d_min
                )
                xs.append(x)
                ys.append(y)
                xs_up.append(x_up)
                ys_up.append(y_up)
                xs_down.append(x_down)
                ys_down.append(y_down)

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
        plot_handles = []
        plot_labels = []

        if self.Cell_ref:  # reference cell chosen!
            s_ref = ax_askew.scatter(
                *tr(x_ref, y_ref), marker="$\u25EF$", c="k", label="ref", alpha=1
            )
            for i, domain in enumerate(self.domains_list):
                s = ax_askew.scatter(
                    *tr(xs_conv[i], ys_conv[i]),
                    marker="o",
                    label=f"domian_{i+1}",
                    c=f"C{i}",
                    alpha=marker_alpha,
                )
                s_up = ax_askew.scatter(
                    *tr(xs_conv_up[i], ys_conv_up[i]),
                    marker="^",
                    label=f"domian_{i+1}",
                    c=f"C{i}",
                    alpha=marker_alpha,
                )
                s_down = ax_askew.scatter(
                    *tr(xs_conv_down[i], ys_conv_down[i]),
                    marker="v",
                    label=f"domian_{i+1}",
                    c=f"C{i}",
                    alpha=marker_alpha,
                )

                h, l = ax_askew.get_legend_handles_labels()
                plot_handles.append((h[1 + i * 3], h[2 + i * 3], h[3 + i * 3]))
                plot_labels.append((l[1 + i * 3]))

            # legned and label for ref
            plot_handles.append((h[0]))
            plot_labels.append((l[0]))

            ax_askew.legend(
                handles=plot_handles,
                labels=plot_labels,
                handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)},
                loc=1,
            )
        else:  # No reference cell
            for i, domain in enumerate(self.domains_list):
                s = ax_askew.scatter(
                    *tr(xs[i], ys[i]),
                    marker="o",
                    label=f"domian_{i+1}",
                    c=f"C{i}",
                    alpha=marker_alpha,
                )
                s_up = ax_askew.scatter(
                    *tr(xs_up[i], ys_up[i]),
                    marker="^",
                    label=f"domian_{i+1}",
                    c=f"C{i}",
                    alpha=marker_alpha,
                )
                s_down = ax_askew.scatter(
                    *tr(xs_down[i], ys_down[i]),
                    marker="v",
                    label=f"domian_{i+1}",
                    c=f"C{i}",
                    alpha=marker_alpha,
                )

                h, l = ax_askew.get_legend_handles_labels()
                plot_handles.append((h[0 + i * 3], h[1 + i * 3], h[2 + i * 3]))
                plot_labels.append((l[0 + i * 3]))

            ax_askew.legend(
                handles=plot_handles,
                labels=plot_labels,
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
    pass
