"""PyInstaller entry point for the console tool (glitchlab-cli.exe)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from glitchlab.cli import main
    sys.exit(main())
