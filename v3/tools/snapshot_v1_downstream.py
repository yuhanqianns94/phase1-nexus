"""Retired bootstrap entry point; V3's adapted sources are maintained directly.

Re-copying the old implementation would restore removed prediction behavior
and overwrite the current extraction/profile/target contracts.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This bootstrap tool is retired. Maintain and test the V3 modules directly; "
        "bulk replacement of the adapted downstream implementation is unsupported."
    )
