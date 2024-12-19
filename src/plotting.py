from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURE_WIDTH = 3.0
FIGURE_HEIGHT = 1.875

FONT_SIZE = 11


_font_path = Path().home() / "Documents/lm2/lmroman10-regular.otf"
font_manager.fontManager.addfont(_font_path)
_font = font_manager.FontProperties(fname=_font_path)


# plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Latin Modern Roman"]



# C_DISS_Y_LABEL = "$c_{\\text{diss}}$ [$mg/L$]"
# C_TOT_Y_LABEL = "$c_{\\text{tot}}$ [$mg/L$]"
C_DISS_Y_LABEL = "$c$ [$mg/L$]"
C_TOT_Y_LABEL = "$c_t$ [$mg/L$]"
C_DISS_X_LABEL = "$t$ [$\\text{days}$]"
C_TOT_X_LABEL = "$x$ [$m$]"
R_X_LABEL = "$c$ [$mg/L$]"
R_Y_LABEL = "$R$ [-]"


def set_retardation_axes_stuff(ax: plt.Axes, xticks=None, yticks=None, set_xlabel=False, set_ylabel=False):
    if xticks is None:
        # xticks = ticker.FixedLocator([0.0, 0.5, 1.0])
        xticks = ticker.FixedLocator([0.0, 0.5, 1.0, 1.5])
    if yticks is None:
        yticks = ticker.MaxNLocator(3)
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)

    if set_xlabel:
        ax.set_xlabel(R_X_LABEL)
    if set_ylabel:
        ax.set_ylabel(R_Y_LABEL)


def set_concentration_axes_stuff(ax: plt.Axes, xticks=None, yticks=None, core="2", set_xlabel=False, set_ylabel=False):
    if xticks is None:
        xticks = (
            ticker.FixedLocator([0, 20, 40])
            if core != "2B"
            else ticker.FixedLocator([0, 0.05, 0.1])
        )
    if yticks is None:
        yticks = (
            ticker.FixedLocator([0, 0.005 / 2, 0.005])
            if core != "2B"
            else ticker.FixedLocator([0, 0.3, 0.6])
        )
    ax.xaxis.set_major_locator(xticks)
    ax.yaxis.set_major_locator(yticks)

    if set_xlabel:
        # ax.set_xlabel("Time [days]" if core != "2B" else "Depth [m]")
        ax.set_xlabel(C_DISS_X_LABEL if core != "2B" else C_TOT_X_LABEL)
    if set_ylabel:
        ax.set_ylabel(
            C_DISS_Y_LABEL
            if core != "2B"
            else C_TOT_Y_LABEL
        )
        # ax.set_ylabel(
        #     "Tailwater concentration [$mg/L$]"
        #     if core != "2B"
        #     else "TCE concentration [$mg/L$]"
        # )


def savefig(fig: plt.Figure, path, tight=True, **kwargs):
    path = Path(path)
    if tight:
        fig.tight_layout()

    for suffix in "png", "svg", "pdf":
        folder = path.parent / suffix
        folder.mkdir(exist_ok=True, parents=True)
        fig.savefig((folder / path.name).with_suffix("." + suffix), **kwargs)

