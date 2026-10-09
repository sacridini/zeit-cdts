# Contributing to Zeit

We welcome contributions from the community. Whether you are fixing bugs, improving documentation, or adding new features, your help is appreciated.

## Development Setup

To start contributing to the Zeit codebase, you will need to set up a local development environment. 

### Prerequisites

* Python 3.9+
* A C++ Compiler (GCC, Clang, or MSVC) for the core engine.

### Setup Instructions

1. **Fork and Clone:** Fork the repository on GitHub and clone your fork locally.
   ```bash
   git clone https://github.com/YOUR_USERNAME/zeit-cdts.git
   cd zeit-cdts
   ```

2. **Install in Editable Mode with Dev Dependencies:** Install the package so that changes to the Python code are immediately reflected without needing to reinstall. 
   ```bash
   pip install -e .[dev]
   ```
   *Note: Any changes made to the C++ source files (`src/*.cpp`) will require you to re-run the `pip install -e .` command to trigger a recompilation.*

## Testing

We use `pytest` for running our test suite. Ensure that all tests pass before submitting a pull request.

To run the tests with coverage reporting, execute:

```bash
pytest --cov=zeit tests/
```

## Building the docs locally

Documentation lives in `docs/` and is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):

```bash
pip install -r docs/requirements.txt
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # link and anchor problems are reported as warnings
```

### API signatures

Every entry in `docs/api/*.md` starts with a marker such as `<!-- sig: zeit.metrics.extract_events -->`. The signature block under it is generated from the code, and the parameter table is checked against it:

```bash
python docs/scripts/sync_api.py           # regenerate signature blocks
python docs/scripts/sync_api.py --check   # report undocumented or unknown parameters, wrong defaults
```

Run it whenever you add, rename or change a parameter.

### Figures

The figures in `docs/assets/figures/` are produced by running Zeit itself:

```bash
python docs/scripts/make_figures.py              # all figures
python docs/scripts/make_figures.py ccdc_fit     # just one
```

Synthetic figures need only Zeit. The real-data figures (LandTrendr, Mann-Kendall, SNIC, SOM) need the Rondônia annual NDVI stack (Landsat NDVI composites for 1985–2024, exported from [LT-GEE](https://github.com/eMapR/LT-GEE) on Google Earth Engine); set `ZEIT_DOCS_RONDONIA` to its path. Set `ZEIT_DOCS_FONT_DIR` to a folder with the Inter font files to match the site's typeface.

### Writing style

- Start each tutorial with a one-sentence lead, the "at a glance" box and a real result figure.
- Explain what the method is for before how it works, and how it works before the code.
- Keep code examples runnable against the current API; prefer real outputs to invented ones.
- Put validation details and implementation notes in a collapsible `??? info` block, and link to [Benchmarks](../benchmarks/index.md) for the full comparison.

## Pull Request Process

1. Create a new branch for your feature or bug fix.
2. Ensure your code follows professional Python standards.
3. Add tests for any new functionality in the `tests/` directory.
4. Update documentation if necessary.
5. Submit a pull request on GitHub, clearly describing the changes and referencing any related issues.
