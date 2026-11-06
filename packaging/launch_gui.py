"""PyInstaller entry point for the windowed application (GlitchLab.exe)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from glitchlab.app import main
    sys.exit(main())
