"""Utility functions for MESHFlow."""
def is_int(s: str) -> bool:
    try:
        int(s) # Uses base 10 unless prefixes (0x, 0b, 0o) appear
        return True
    except ValueError:
        return False