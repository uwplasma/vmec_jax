# Bundled Runtime Resources

This package contains small files that must be available after `pip install
vmex`.

Keep this directory small. Large WOUTs, MGRID files, generated optimization
outputs, and validation fixtures belong in release assets fetched by
`tools/fetch_assets.py`, not in the Python package.

User-facing examples and input decks live under `examples/data/`. This package
directory is only for tiny resources required by installed command-line helpers
such as `vmec --test`.
