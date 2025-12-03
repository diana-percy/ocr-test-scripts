"""
Shared utilities for OCR test scripts
"""

import os

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def resolve_input_path(filename):
    """Resolve input file path - check resources folder if not found directly"""
    # If it's an absolute path or exists as-is, use it
    if os.path.isabs(filename) or os.path.exists(filename):
        return filename
    
    # Check in resources folder
    resources_path = os.path.join(RESOURCES_DIR, filename)
    if os.path.exists(resources_path):
        return resources_path
    
    # Return original (will fail with appropriate error later)
    return filename


def resolve_output_path(filename):
    """Resolve output file path - use output folder"""
    # If it's an absolute path, use it as-is
    if os.path.isabs(filename):
        return filename
    
    # Otherwise, put it in the output folder
    return os.path.join(OUTPUT_DIR, filename)

