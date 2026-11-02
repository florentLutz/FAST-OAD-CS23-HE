# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import click

from .power_train_builder_launcher import PowertrainBuilderLauncher


@click.command()
@click.option("--port", default=5007, show_default=True, help="TCP port of the Bokeh server.")
@click.option("--address", default="localhost", show_default=True, help="Server bind address.")
def main(port, address):
    """Launch the powertrain builder GUI in the default web browser."""
    PowertrainBuilderLauncher.launch(port=port, address=address)


if __name__ == "__main__":
    main()
