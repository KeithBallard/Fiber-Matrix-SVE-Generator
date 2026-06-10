# Offline And GitHub Pages Documentation

`build_docs.py` supports both a live preview server and a static offline build.

## Live Preview

```bash
python build_docs.py --serve
```

Open `http://127.0.0.1:8000` in a browser.

## Static Offline Build

```bash
python build_docs.py --offline
```

This creates a static site in the `site/` directory. The generated HTML uses
relative links, so it can be opened locally from:

```text
site/index.html
```

The same `site/` directory is also deployable to GitHub Pages.

## GitHub Pages Deployment

One simple deployment path is:

```bash
python build_docs.py --offline
python -m mkdocs gh-deploy
```

Alternatively, upload the generated `site/` directory as a GitHub Pages artifact
or publish it from a deployment branch.
