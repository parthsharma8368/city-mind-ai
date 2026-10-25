"""Builds docs/PROJECT_REPORT_FOR_PROFESSOR.pdf from report.html.tmpl (+ fig/*.png) with headless Chrome.

  .venv/bin/python docs/professor_report/make_figures.py      # figures (needs artifacts/eval10_run/report)
  .venv/bin/python docs/professor_report/build_report.py      # HTML + PDF
{{c:path:line}} -> link to that file/line in the repository; {{b:Ch.3 p.44}} -> book tag. Every code reference is checked to exist.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
BASE = "https://github.com/parthsharma8368/city-mind-ai/blob/feat/peak-hour-mechanics-fix/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT_PDF = ROOT / "docs" / "PROJECT_REPORT_FOR_PROFESSOR.pdf"
verbose = "--list" in sys.argv
problems: list[str] = []


def code(m: re.Match) -> str:
    path, line = m.group(1), int(m.group(2))
    f = ROOT / path
    if not f.is_file():
        problems.append(f"missing file {path}")
    else:
        lines = f.read_text(errors="replace").splitlines()
        if line > len(lines):
            problems.append(f"{path}:{line} beyond {len(lines)} lines")
        elif verbose:
            print(f"{path}:{line}  {lines[line - 1].strip()[:90]}")
    label = f"{path}:{line}" if line > 1 else path
    href = f"{BASE}{path}" + (f"#L{line}" if line > 1 else "")
    return f'<a class="c" href="{href}">{label}</a>'


html = (HERE / "report.html.tmpl").read_text()
html = re.sub(r"\{\{c:([^:}]+):(\d+)\}\}", code, html)
html = re.sub(r"\{\{b:([^}]+)\}\}", lambda m: f'<span class="bk">BOOK {m.group(1)}</span>', html)
(HERE / "report.html").write_text(html)
if problems:
    sys.exit("PROBLEMS:\n" + "\n".join(problems))
if "--no-pdf" in sys.argv:
    sys.exit(0)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                (HERE / "report.html").as_uri()], check=True, capture_output=True)
print("wrote", OUT_PDF)
