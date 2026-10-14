import runpy
from pathlib import Path

import pytest

from twinindex import Cell, Domains

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


def test_cell_plots():
    cell = Cell(spgr="P21/c", lattice_params=(9.185, 5.8294, 7.9552, 100.79))
    cell.plot_cell()
    cell.plot_peaks(projection="H,0,L", del_Q=0.2, d_min=0.5)


def test_domains_mirror():
    cell = Cell(spgr="P21/c", lattice_params=(9.185, 5.8294, 7.9552, 100.79))
    domains = Domains(cell)
    domains.make_twin_mirror(plane=[1, 0, 0], origin=[-1, -1, -1])
    domains.plot_domains()
    domains.plot_peaks(projection="H,0,L", del_Q=0.1, d_min=1)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(example):
    runpy.run_path(str(example), run_name="__main__")
