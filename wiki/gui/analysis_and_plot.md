# Analysis and plots

This service currently contains only a helper function to plot the top view of the aircraft geometry.

The function is `aircraft_geometry_plot` in `analysis_and_plots.py`.

## Signature

```python
aircraft_geometry_plot(
    aircraft_file_path: str,
    name: str = "",
    fig=None,
    plot_nacelle: bool = True,
    file_formatter=None,
) -> go.FigureWidget
```

| Argument | Role                                                                                                                                     |
|---|------------------------------------------------------------------------------------------------------------------------------------------|
| `aircraft_file_path` | Path of the FAST-OAD data file (input or output file of a design run).                                                                   |
| `name` | Label of the trace in the legend. Useful when comparing several designs.                                                                 |
| `fig` | An existing Plotly figure. If given, the new design is added to it, so several designs or the aircraft top-view image can be superposed. |
| `plot_nacelle` | Turns the drawing of nacelles and propellers on or off.                                                                                  |
| `file_formatter` | Formatter describing the data file format. If `None`, the default FAST-OAD format is assumed.                                            |

The function returns a `plotly.graph_objects.FigureWidget`, which is displayed interactively in a Jupyter notebook.

## Overview of the process

1. Read the geometry variables from the output file.
2. Build the outline of the wing, horizontal tail and fuselage in their own local frames.
3. Move the wing and tail into the common aircraft frame.
4. Mirror the half-aircraft to obtain the full top view and add it as one trace.
5. Build and add the nacelle and propeller outline, depending on the propulsion layout.
6. Apply the layout (equal axis scaling, titles, legend) and return the figure.

## Step by step

### 1. Reading the data

```python
variables = VariableIO(aircraft_file_path, file_formatter).read()
```

`VariableIO` (from FAST-OAD) parses the file into a collection of variables indexed by their full name, for example 
`data:geometry:wing:root:chord`. Each variable holds an array in `.value`. All the quantities used here are scalars, so 
the code takes `.value[0]`. The nacelle positions are the exception: they are kept as full arrays because there can be 
several nacelles.

### 2. Plot frame

The plot uses a **swapped frame**:

- Plot horizontal axis = aircraft **y** (spanwise).
- Plot vertical axis = aircraft **x** (longitudinal, from the nose, positive towards the tail).

This is why the final `Scatter` is called with `x=y, y=x`. The nose is at the bottom of the plot and the tail at the top.

Every outline is built for **one half** of the aircraft (y ≥ 0) and mirrored at the end.

### 3. Wing outline

Variables used: root, kink and tip spanwise positions (`wing:root:y`, `wing:kink:y`, `wing:tip:y`), the three chords, 
and the local leading-edge x of the kink and tip.

The wing is a polygon traced as a closed loop (nine points). It goes along the **leading edge** from root to kink to 
tip, then back along the **trailing edge** from tip to kink to root, and closes at the centerline.

- Leading-edge x values: `0`, `kink_leading_edge_x`, `tip_leading_edge_x`.
- Trailing-edge x values: leading-edge x plus the local chord at the tip and kink, then `root_chord` at the root.

The wing is first defined in a **local frame** whose origin is the root leading edge.

### 4. Horizontal tail outline

Variables used: root chord, tip chord, span and `sweep_0` (sweep angle in degrees).

- Half span is `ht_span / 2`.
- The tip leading-edge offset is `half_span · tan(sweep_0)`, with the angle converted from degrees to radians.
- The four corners are built around a reference at the root quarter-chord: the leading edge is at `-0.25 · chord` and 
the trailing edge at `+0.75 · chord`. The tip corners are offset in the same way using the tip chord.

### 5. Fuselage outline

Variables used: maximum width, total length, front length and rear length.

The half-outline is a simplified six-point profile:

| Point | x | y (half-width) |
|---|---|---|
| Nose centre | 0 | 0 |
| Nose edge | 0 | width / 4 |
| End of front section | front length | width / 2 |
| Start of rear section | length − rear length | width / 2 |
| Tail edge | length | width / 4 |
| Tail centre | length | 0 |

So the nose and tail are truncated with a blunt end of half the maximum half-width, and the middle section has constant 
width.

### 6. Placing wing and tail in the aircraft frame

The wing and tail are defined locally and must be moved so that they sit correctly relative to the fuselage:

```python
x_wing = x_wing + wing_25mac_x - 0.25 * wing_mac_length - local_wing_mac_le_x
x_ht   = x_ht   + wing_25mac_x + ht_distance_from_wing - local_ht_25mac_x
```

- **Wing:** `wing:MAC:at25percent:x` is the aircraft-frame position of the wing quarter mean aerodynamic chord (MAC). 
Subtracting `0.25 · MAC length` gives the MAC leading edge. Subtracting the local MAC leading-edge x then gives the 
position of the wing root leading edge. The whole wing polygon is shifted by that amount.
- **Tail:** The tail quarter-MAC is placed at the wing quarter-MAC plus the tail lever arm 
(`horizontal_tail:MAC:at25percent:x:from_wingMAC25`). The local tail quarter-MAC x is subtracted so that the local 
origin matches.

The y coordinates need no shift, since both surfaces start on the centerline.

### 7. Assembling and mirroring

```python
x = np.concatenate((x_fuselage, x_wing, x_ht))
y = np.concatenate((y_fuselage, y_wing, y_ht))
y = np.concatenate((-y, y))
x = np.concatenate((x, x))
```

- The three outlines are chained into a **single trace**.
- Each outline starts and ends at `y = 0`. The connecting segments between outlines therefore lie on the symmetry axis 
and are not visible.
- Mirroring: `-y` is concatenated with `y` (left side then right side), and `x` is duplicated.

If no `fig` was given, a new `go.Figure()` is created. The trace is added with `mode="lines+markers"`, the given `name`,
and `showlegend=True`.

### 8. Nacelles and propellers

Variables used: `propulsion:engine:layout`, nacelle width and length, propeller diameter, and the nacelle position 
arrays `nacelle:x` and `nacelle:y`.

Two propulsion layouts are handled. For any other layout value, the nacelle outline is empty and nothing is drawn.

**Layout 1 (propellers on the wing):**

- The base shape is a rectangle of length `nac_length` and width `nac_width`, plus a vertical segment of length 
`prop_diam` representing the propeller disc.
- For each nacelle position `(x, y)`, the left nacelle is shifted by `+y` and the right one is its mirror 
(`-shape - y`). In the longitudinal direction, `x_nacelle = x_local - shape_x`.
- A color is picked at random from Plotly's default palette (`SystemRandom`). The left and right nacelles share the same
color and the same `legendgroup`, so one legend click toggles both.

**Layout 3 (fuselage-mounted):**

- The shape is the same, but shifted rearwards by the propeller depth (`propeller:depth`).
- The nacelle half-width is clipped to `fuselage_max_width / 4` with `max`/`min`, so the nacelle does not stick out of 
the fuselage.
- The shape is plotted directly as a single trace, with `name + " nacelle + propeller"`.

If `plot_nacelle=False`, this whole section is computed but not drawn.

### 9. Layout and return

```python
fig.layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))
fig = go.FigureWidget(fig)
fig.update_layout(title_text="Aircraft Geometry", ...)
```

- Linking the y axis to the x axis with ratio 1 gives an **equal aspect ratio**, so the aircraft is not distorted.
- The figure is converted to a `FigureWidget` (interactive in Jupyter).
- The title is centered, the axes are labeled `y` and `x`, and the legend is placed in the top-right corner.

## Usage example

```python
from analysis_and_plots import aircraft_geometry_plot

fig = aircraft_geometry_plot("data/design_1.xml", name="Design 1")
fig = aircraft_geometry_plot("data/design_2.xml", name="Design 2", fig=fig)
fig.show()
```

## Points to be aware of

- **Only the first nacelle pair is drawn for layout 1.** The `add_trace` calls are inside the `if show_legend:` block, 
and `show_legend` is set to `False` after the first iteration. The aim is probably to avoid duplicate legend entries, 
but later nacelles are skipped entirely. Moving the trace creation outside the `if`, and using `showlegend=show_legend` 
instead, would draw all nacelles.
- **Layout 3 ignores `nacelle:x` and `nacelle:y`.** The shape is drawn at its local position (shifted only by the 
propeller depth), not at the position given in the data file.
- **Overwriting `fig.layout`.** The `go.Layout(...)` assignment replaces any layout already present on a figure passed 
through `fig`. The later `update_layout` call then re-applies title, axis labels and legend.
- **Simplified fuselage.** The fuselage is a six-point outline, not a detailed shape.
- **Random nacelle color.** The color is not reproducible between runs.