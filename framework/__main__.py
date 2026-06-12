"""Allow running: python -m framework"""
from .framework_runner import run_framework

if __name__ == "__main__":
    run_framework(force_fetch=True)
