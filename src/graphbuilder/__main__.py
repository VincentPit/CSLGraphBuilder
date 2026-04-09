"""
GraphBuilder - Main module entry point.

Allows running GraphBuilder as a module: python -m graphbuilder
"""

from .cli.main import cli

if __name__ == '__main__':
    cli()
