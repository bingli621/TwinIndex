import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def headless_figures(monkeypatch):
    # examples call plt.show(), which has no window to open under Agg
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")
