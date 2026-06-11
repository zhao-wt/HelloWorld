"""
bear/export_pdf.py — render the model tech-doc markdown files to PDF.

Pipeline: markdown -> HTML (math protected, MathJax for LaTeX, local PNGs)
-> headless Chrome --print-to-pdf. No pandoc / LaTeX toolchain required.

Run:  python -m bear.export_pdf
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import markdown

_DOCS_DIR = Path(__file__).resolve().parent / "docs"

_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

_CSS = """
<style>
  @page { size: A4; margin: 18mm 16mm; }
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #18211b; font-size: 11pt; line-height: 1.45; }
  h1 { color: #173f2a; font-size: 20pt; border-bottom: 3px solid #b68a35;
       padding-bottom: 6px; }
  h2 { color: #173f2a; font-size: 15pt; margin-top: 1.4em;
       border-bottom: 1px solid #dce3dd; padding-bottom: 3px; }
  h3 { color: #24563a; font-size: 12.5pt; margin-top: 1.1em; }
  table { border-collapse: collapse; width: 100%; margin: 0.8em 0;
          font-size: 9.5pt; }
  th, td { border: 1px solid #dce3dd; padding: 5px 8px; text-align: left; }
  th { background: #f3f4f6; }
  td:not(:first-child), th:not(:first-child) { text-align: right; }
  img { max-width: 100%; display: block; margin: 0.6em auto; }
  code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
         font-size: 9.5pt; }
  em { color: #5d675f; }
  h2, h3, table, img { page-break-inside: avoid; }
</style>
"""

_MATHJAX = """
<script>
  window.MathJax = {
    tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] },
    svg: { fontCache: 'global' }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Replace $$...$$ and $...$ with placeholders so markdown leaves TeX intact."""
    store: list[str] = []

    def repl(m: re.Match) -> str:
        store.append(m.group(0))
        return f"@@MATH{len(store)-1}@@"

    text = re.sub(r"\$\$.*?\$\$", repl, text, flags=re.DOTALL)   # display first
    text = re.sub(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", repl, text)  # inline
    return text, store


def _restore_math(html: str, store: list[str]) -> str:
    """Put the raw TeX back, HTML-escaping <, >, & so the markup stays valid."""
    for i, tex in enumerate(store):
        safe = tex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = html.replace(f"@@MATH{i}@@", safe)
    return html


def md_to_html(md_path: Path) -> Path:
    raw = md_path.read_text()
    protected, store = _protect_math(raw)
    body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = _restore_math(body, store)

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        + _CSS + _MATHJAX
        + "</head><body>" + body + "</body></html>"
    )
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html)
    return html_path


def html_to_pdf(html_path: Path) -> Path:
    pdf_path = html_path.with_suffix(".pdf")
    cmd = [
        _CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return pdf_path


if __name__ == "__main__":
    docs = ["bear_model_tech_doc.md", "correction_model_tech_doc.md"]
    for name in docs:
        md_path = _DOCS_DIR / name
        if not md_path.exists():
            print(f"  ! {name} not found — run `python -m bear.make_docs` first.")
            continue
        html_path = md_to_html(md_path)
        pdf_path = html_to_pdf(html_path)
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  {name}  ->  {pdf_path.name}  ({size_kb:.0f} KB)")
    print(f"\nPDFs written to {_DOCS_DIR}")
