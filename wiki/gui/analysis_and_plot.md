# Analysis and plots

This service currently contains only a helper function to plot the top view of the aircraft geometry.

---

## aircraft_geometry_plot

```python
aircraft_geometry_plot(aircraft_file_path, name='', fig=None, plot_nacelle=True, file_formatter=None)
```

Plot the top view of the aircraft geometry.

The plot shows the fuselage, the wing, the horizontal tail and, optionally, the nacelles and propellers. Different designs can be superposed by providing an existing figure, and each design can be given a name.

---

### Parameters

- **aircraft_file_path** : *str*

  Path of the data file.

- **name** : *str, optional*

  Name to give to the trace added to the figure (default: `''`).

- **fig** : *plotly.graph_objects.Figure, optional*

  Existing figure to which the plot is added. If `None` (default), a new figure is created.

- **plot_nacelle** : *bool, optional*

  Turns the plotting of the nacelles and propellers on or off (default: `True`).

- **file_formatter** : *IVariableIOFormatter, optional*

  Formatter that defines the format of the data file. If not provided, the default format is assumed.

---

### Returns

- **figure** : *plotly.graph_objects.FigureWidget*

  Interactive figure of the aircraft top view. When `fig` is given, it also contains the traces already present in `fig`.

---

### See also

- [`fastoad.io.VariableIO`](https://fast-oad.readthedocs.io/en/v0.5.4-beta/api/fastoad.io.xml.variable_io_standard.html): reader used to load the data file.

---

### Notes

The plot is rotated compared to the aircraft frame. The horizontal axis is the aircraft `y` (span) and the vertical axis is the aircraft `x` (longitudinal), so the nose is at the bottom and the tail at the top. The axes have equal scaling, so the geometry is not distorted.

The plot is built in six steps.

#### 1. Read the data

The variables are read from the data file with `VariableIO`. All variables are scalars (`.value[0]`), except the nacelle positions, which are kept as arrays since there can be several nacelles.

#### 2. Build the local outlines

Each component is defined as a half-outline (`y >= 0`) in its own local frame.

- **Wing:** a closed nine-point polygon. It follows the leading edge from root to kink to tip, then the trailing edge back from tip to kink to root. The trailing-edge x is the leading-edge x plus the local chord. The local origin is the root leading edge.
- **Horizontal tail:** the half span is `span / 2` and the tip leading-edge offset is `half_span * tan(sweep_0)`, with `sweep_0` in degrees. The corners are built around a reference at the root quarter-chord: the leading edge at `-0.25 * chord` and the trailing edge at `+0.75 * chord`.
- **Fuselage:** a six-point profile with a blunt nose and tail of half-width `maximum_width / 4`, and a constant half-width of `maximum_width / 2` between the end of the front section and the start of the rear section.

#### 3. Move the wing and tail to the aircraft frame

```python
x_wing = x_wing + wing_25mac_x - 0.25 * wing_mac_length - local_wing_mac_le_x
x_ht = x_ht + wing_25mac_x + ht_distance_from_wing - local_ht_25mac_x
```

The wing is positioned from the aircraft-frame position of its quarter mean aerodynamic chord (MAC). The tail quarter-MAC is placed behind the wing quarter-MAC by the tail lever arm. No shift is needed in `y`, since both surfaces start on the centerline.

#### 4. Assemble and mirror

The three outlines are concatenated into a single trace. Each one starts and ends at `y = 0`, so the segments linking them lie on the symmetry axis and are not visible. The half-aircraft is then mirrored (`y -> -y`) and added to the figure with `lines+markers`.

#### 5. Add the nacelles and propellers

If `plot_nacelle` is `True`, the shape depends on `data:geometry:propulsion:engine:layout`.

- **`1.0` (propellers on the wing):** a rectangle of the nacelle length and width, plus a segment of the propeller diameter for the propeller disc. For each nacelle position, the left nacelle is shifted by the nacelle `y` and the right one is its mirror. Both share a random color from the Plotly default palette and a common legend group.
- **`3.0` (fuselage-mounted):** the same shape shifted rearwards by the propeller depth, with its half-width limited to `maximum_width / 4` so that it stays within the fuselage.
- **Any other value:** no nacelle is drawn.

#### 6. Apply the layout

The layout is applied (equal axis scaling, centered title `Aircraft Geometry`, axis titles `y` and `x`, legend in the top-right corner) and the figure is converted to a `FigureWidget`.

#### Variables read

All names are prefixed with `data:geometry:`.

| Component | Variables |
|---|---|
| Wing | `wing:root:y`, `wing:kink:y`, `wing:tip:y`, `wing:root:chord`, `wing:kink:chord`, `wing:tip:chord`, `wing:kink:leading_edge:x:local`, `wing:tip:leading_edge:x:local`, `wing:MAC:at25percent:x`, `wing:MAC:length`, `wing:MAC:leading_edge:x:local` |
| Horizontal tail | `horizontal_tail:root:chord`, `horizontal_tail:tip:chord`, `horizontal_tail:span`, `horizontal_tail:sweep_0`, `horizontal_tail:MAC:at25percent:x:local`, `horizontal_tail:MAC:at25percent:x:from_wingMAC25` |
| Fuselage | `fuselage:maximum_width`, `fuselage:length`, `fuselage:front_length`, `fuselage:rear_length` |
| Propulsion | `propulsion:engine:layout`, `propulsion:nacelle:width`, `propulsion:nacelle:length`, `propulsion:nacelle:x`, `propulsion:nacelle:y`, `propeller:diameter`, `propeller:depth` (layout `3.0` only) |

---

### Examples

```python
from analysis_and_plots import aircraft_geometry_plot
fig = aircraft_geometry_plot("data/design_1.xml", name="Design 1")
fig.show()
```

Superpose a second design on the same figure:

```python
fig = aircraft_geometry_plot("data/design_2.xml", name="Design 2", fig=fig)
```

Plot the airframe only:

```python
fig = aircraft_geometry_plot("data/design_1.xml", plot_nacelle=False)
```