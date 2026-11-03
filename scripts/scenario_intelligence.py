"""Scenario intelligence command-line entry point (see ``--help`` for subcommands)."""

import sys

from scenarioforge.scenario_intelligence.cli import main

if __name__ == "__main__":
    sys.exit(main())
