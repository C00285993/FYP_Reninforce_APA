#!/usr/bin/env python
"""Shortcut entry point — run the interactive assistant with: python assistant.py"""
import sys
import os

# Ensure the project root is on the path so all package imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.assistant import main

if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n  [Session ended — goodbye.]")
