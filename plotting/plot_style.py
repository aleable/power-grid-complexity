from contextlib import contextmanager

import cycler
import matplotlib.pyplot as plt

COLORS = [
    "#0d8fe5",
    "#eb3f00",
    "#92c700",
    "#f272f0",
]

STYLE_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 10,
    "figure.figsize": (3.5, 3),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": "silver",
    "grid.linestyle": "dotted",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "axes.titlesize": 10,
    "axes.grid": False,
    "axes.prop_cycle": cycler.cycler("color", COLORS),
    # Ensure math text uses Helvetica light
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    # For SVG output as real text
    "svg.fonttype": "none",
}


@contextmanager
def plot_style(color_scheme=COLORS, style_params=STYLE_PARAMS):

    current_style = STYLE_PARAMS.copy()

    if style_params:
        current_style.update(style_params)

    if color_scheme:
        if isinstance(color_scheme, str):
            if color_scheme.lower() == "colors":
                colors = COLORS
            else:
                raise ValueError(f"Unknown color scheme: {color_scheme}")
        else:
            colors = color_scheme

        current_style["axes.prop_cycle"] = cycler.cycler("color", colors)

    with plt.style.context(current_style):
        yield


@staticmethod
def lighten_color(color, amount=0.5):
    """
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
