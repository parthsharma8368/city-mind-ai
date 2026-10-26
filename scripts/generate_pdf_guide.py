"""
generate_pdf.py — Converts LEARNING_GUIDE_ALL_POLICIES.md into a publication-grade PDF via Google Chrome.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path("/Users/parthsharma/Desktop/FInal Try")
MD_PATH = ROOT / "docs" / "LEARNING_GUIDE_ALL_POLICIES.md"
HTML_PATH = ROOT / "docs" / "learning_guide_print.html"
PDF_PATH = ROOT / "docs" / "CityMind_AI_Policy_Learning_Guide.pdf"

md_content = MD_PATH.read_text(encoding="utf-8")

# Basic conversion of markdown elements to HTML
def md_to_html(text: str) -> str:
    lines = text.split("\n")
    html_lines = []
    in_code_block = False
    code_lang = ""
    code_buffer = []
    in_table = False
    table_buffer = []

    for line in lines:
        # Code blocks
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = line[3:].strip()
                code_buffer = []
                continue
            else:
                in_code_block = False
                escaped_code = (
                    "\n".join(code_buffer)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                html_lines.append(f'<pre class="code-block {code_lang}"><code>{escaped_code}</code></pre>')
                continue

        if in_code_block:
            code_buffer.append(line)
            continue

        # Tables
        if "|" in line and not in_code_block:
            if not in_table:
                in_table = True
                table_buffer = [line]
                continue
            else:
                table_buffer.append(line)
                continue
        elif in_table:
            in_table = False
            # Render table
            rows = []
            for t_idx, t_line in enumerate(table_buffer):
                if re.match(r"^\s*\|?[-:| ]+\|?\s*$", t_line):
                    continue  # delimiter line
                cols = [c.strip() for c in t_line.split("|")[1:-1]]
                tag = "th" if t_idx == 0 else "td"
                cells = "".join(f"<{tag}>{c}</{tag}>" for c in cols)
                rows.append(f"<tr>{cells}</tr>")
            html_lines.append(f'<table class="styled-table">{"".join(rows)}</table>')
            table_buffer = []

        # Empty lines
        if not line.strip():
            html_lines.append('<div class="spacer"></div>')
            continue

        # Headings
        if line.startswith("# "):
            html_lines.append(f'<h1 class="h1-title">{line[2:].strip()}</h1>')
        elif line.startswith("## "):
            html_lines.append(f'<h2 class="h2-title">{line[3:].strip()}</h2>')
        elif line.startswith("### "):
            html_lines.append(f'<h3 class="h3-title">{line[4:].strip()}</h3>')
        elif line.startswith("#### "):
            html_lines.append(f'<h4 class="h4-title">{line[5:].strip()}</h4>')
        elif line.startswith("> "):
            callout = line[2:].strip()
            html_lines.append(f'<div class="callout"><p>{callout}</p></div>')
        elif line.startswith("* ") or line.startswith("- "):
            item = line[2:].strip()
            html_lines.append(f'<li class="list-item">{item}</li>')
        else:
            html_lines.append(f'<p class="para">{line.strip()}</p>')

    raw_html = "\n".join(html_lines)

    # Inline formatting (bold, italic, code, links)
    raw_html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", raw_html)
    raw_html = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", raw_html)
    raw_html = re.sub(r"`([^`]+)`", r'<code class="inline-code">\1</code>', raw_html)
    raw_html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', raw_html)

    return raw_html

body_html = md_to_html(md_content)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CityMind AI — Policy Learning Guide</title>
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-center {{
      content: counter(page);
    }}
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1f2937;
    background-color: #ffffff;
    line-height: 1.6;
    font-size: 10.5pt;
    margin: 0;
    padding: 0;
  }}

  .cover-banner {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: #ffffff;
    padding: 32px 28px;
    border-radius: 8px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }}

  .cover-banner h1 {{
    margin: 0 0 8px 0;
    font-size: 20pt;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
  }}

  .cover-banner h2 {{
    margin: 0 0 16px 0;
    font-size: 13pt;
    font-weight: 400;
    color: #93c5fd;
  }}

  .cover-meta {{
    font-size: 8.5pt;
    color: #cbd5e1;
    display: flex;
    gap: 20px;
  }}

  .h1-title {{
    font-size: 15pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 4px;
    margin-top: 28px;
    margin-bottom: 12px;
    page-break-after: avoid;
  }}

  .h2-title {{
    font-size: 12.5pt;
    font-weight: 600;
    color: #1e40af;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}

  .h3-title {{
    font-size: 11pt;
    font-weight: 600;
    color: #374151;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  .para {{
    margin: 0 0 8px 0;
    text-align: justify;
  }}

  .list-item {{
    margin-bottom: 4px;
    margin-left: 18px;
  }}

  .callout {{
    background-color: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 9.5pt;
    color: #166534;
  }}

  .inline-code {{
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 1px 5px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8.5pt;
    border: 1px solid #e2e8f0;
  }}

  .code-block {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 12px 14px;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8pt;
    line-height: 1.45;
    overflow-x: auto;
    margin: 10px 0;
    page-break-inside: avoid;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
  }}

  .styled-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}

  .styled-table th {{
    background-color: #1e293b;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 7px 10px;
    border: 1px solid #334155;
  }}

  .styled-table td {{
    padding: 6px 10px;
    border: 1px solid #cbd5e1;
  }}

  .styled-table tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  .spacer {{
    height: 4px;
  }}

  a {{
    color: #2563eb;
    text-decoration: none;
  }}

  @media print {{
    body {{
      font-size: 9.5pt;
    }}
    .h1-title {{
      page-break-before: auto;
    }}
  }}
</style>
</head>
<body>

<div class="cover-banner">
  <h1>CityMind AI: Deep Reinforcement Learning for Traffic Signal Control</h1>
  <h2>Comprehensive Theoretical Guide & Codebase Architecture Manual</h2>
  <div class="cover-meta">
    <span><strong>Author:</strong> CityMind AI Research Engineering</span>
    <span><strong>Scenario:</strong> Ingolstadt InTAS (98 TLS)</span>
    <span><strong>Frameworks:</strong> Eclipse SUMO, PyTorch, Stable-Baselines3</span>
  </div>
</div>

{body_html}

</body>
</html>
"""

HTML_PATH.write_text(full_html, encoding="utf-8")
print(f"Wrote HTML to {HTML_PATH}")

# Convert to PDF via headless Google Chrome
chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_bin,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={PDF_PATH}",
    "--no-pdf-header-footer",
    f"file://{HTML_PATH.resolve()}",
]

print("Executing Chrome headless PDF generation...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
if res.returncode == 0:
    print(f"SUCCESS: Generated PDF at {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
else:
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
