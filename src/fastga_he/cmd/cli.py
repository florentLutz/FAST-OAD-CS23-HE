# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""Command Line Interface of FAST-OAD_CS23-HE (``fastga-he``)."""

from importlib.metadata import PackageNotFoundError, packages_distributions, version

import click


def _get_version() -> str:
    """Version of the installed distribution that provides the ``fastga_he`` package."""
    try:
        dist_name = packages_distributions().get("fastga_he", ["fastga_he"])[0]
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_get_version(), "-v", "--version")
def fastga_he_cli():
    """FAST-OAD_CS23-HE main program.

    Standard FAST-OAD commands (gen_conf, eval, optim, ...) remain available
    through the `fastoad` command.
    """


@fastga_he_cli.command(name="powertrain_builder")
@click.option("--port", default=5007, show_default=True, help="TCP port of the Bokeh server.")
@click.option("--address", default="localhost", show_default=True, help="Server bind address.")
def powertrain_builder(port, address):
    """Launch the powertrain builder GUI in the default web browser."""
    # Imported here so that `fastga-he --help` stays fast and does not load Bokeh.
    from fastga_he.gui.power_train_builder import PowertrainBuilderLauncher

    PowertrainBuilderLauncher.launch(port=port, address=address)


if __name__ == "__main__":
    fastga_he_cli()
