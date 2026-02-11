"""Single point place to retrieve the `hazenlib` version.

This module provides a single canonical location for determining the
version string exposed as ``hazenlib.__version__``.

Version resolution strategy
---------------------------

* **Package vs distribution names**
  The importable package is named :mod:`hazenlib`, while the installed
  distribution (the project published on PyPI) is typically named
  ``hazen``. Because a single package can belong to one or more
  distributions, we first ask :func:`importlib.metadata.packages_distributions`
  which distribution name(s) provide the ``hazenlib`` package, and then
  query each distribution for its version.

* **Fallback to ``pyproject.toml``**
  When ``hazenlib`` is imported from a source tree without an installed
  distribution (for example, during development, in a local checkout, or
  in some test environments), querying the distribution metadata can
  fail. In that case, we fall back to reading the version declared in
  the top-level ``pyproject.toml`` under ``[project].version``.

* **``+dev`` suffix**
  If we have to fall back to ``pyproject.toml``, we append a ``\"+dev\"``
  suffix to the version. This indicates that the code is being run from
  a development (source) environment rather than from an installed,
  released distribution. This also avoids accidentally reporting an
  identical version string for both an installed release and a
  development checkout.

* **Order of precedence**
  1. Use the version from the first distribution associated with the
     ``hazenlib`` package whose metadata can be successfully read via
     :func:`importlib.metadata.version`.
  2. If no such distribution is found or all lookups fail, read the
     version from ``pyproject.toml`` and append ``\"+dev\"``.
"""

import importlib.metadata
import pathlib
import tomllib

distributions = importlib.metadata.packages_distributions().get("hazenlib", [])

__version__ = None
for dist_name in distributions:
    try:
        __version__ = importlib.metadata.version(dist_name)
        break
    except importlib.metadata.PackageNotFoundError:
        continue

if __version__ is None:
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse {pyproject_path} as TOML while resolving hazenlib version"
        ) from exc
    __version__ = data["project"]["version"] + "+dev"
