from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Category10
import numpy as np


def create_app():
    """Create the Bokeh playback slider app"""
    # Create sample data - simulating frames of animation
    n_frames = 50
    n_points = 100
    frames_data = []

    for frame in range(n_frames):
        x = np.linspace(0, 4 * np.pi, n_points)
        y = np.sin(x + frame * 0.2)
        frames_data.append({"x": x, "y": y})

    # Initialize with first frame
    source = ColumnDataSource(data=dict(x=frames_data[0]["x"], y=frames_data[0]["y"]))

    # Create plot
    plot = figure(
        title="Playback Slider Example",
        width=600,
        height=400,
        x_range=[0, 4 * np.pi],
        y_range=[-1.5, 1.5],
    )
    plot.line("x", "y", source=source, line_width=2, color="navy")
    plot.circle("x", "y", source=source, size=4, color="red", alpha=0.5)

    # Create slider
    slider = Slider(start=0, end=n_frames - 1, value=0, step=1, title="Frame")

    # Create buttons for playback control
    play_button = Button(label="▶ Play", width=80)
    pause_button = Button(label="⏸ Pause", width=80)
    stop_button = Button(label="⏹ Stop", width=80)

    # Playback state
    playback_state = {"playing": False, "interval": None}

    def update_frame(attr, old, new):
        """Update plot when slider changes"""
        frame_idx = int(slider.value)
        source.data = dict(x=frames_data[frame_idx]["x"], y=frames_data[frame_idx]["y"])

    def play_callback():
        """Start playback"""
        playback_state["playing"] = True
        play_button.disabled = True
        pause_button.disabled = False

        def animate():
            if playback_state["playing"]:
                current = slider.value
                if current < n_frames - 1:
                    slider.value = current + 1
                else:
                    # Loop back to start
                    slider.value = 0

        playback_state["interval"] = curdoc().add_periodic_callback(animate, 100)

    def pause_callback():
        """Pause playback"""
        playback_state["playing"] = False
        play_button.disabled = False
        pause_button.disabled = True

    def stop_callback():
        """Stop playback and reset"""
        playback_state["playing"] = False
        if playback_state["interval"]:
            curdoc().remove_periodic_callback(playback_state["interval"])
            playback_state["interval"] = None
        slider.value = 0
        play_button.disabled = False
        pause_button.disabled = True

    # Connect callbacks
    slider.on_change("value", update_frame)
    play_button.on_click(play_callback)
    pause_button.on_click(pause_callback)
    stop_button.on_click(stop_callback)

    # Initially disable pause button
    pause_button.disabled = True

    # Layout
    controls = row(play_button, pause_button, stop_button)
    layout = column(plot, slider, controls)

    return layout


if __name__ == "__main__":
    from bokeh.server.server import Server
    import webbrowser

    def make_doc(doc):
        layout = create_app()
        doc.add_root(layout)
        doc.title = "Bokeh Playback Slider"

    server = Server({"/": make_doc}, num_procs=1)
    server.start()

    print("Opening Bokeh application on http://localhost:5006/")
    webbrowser.open("http://localhost:5006/")
    server.io_loop.start()
