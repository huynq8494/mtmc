"""Build docs/MTMC.html from docs/MTMC.md.   Run:  uv run --with markdown python docs/build_html.py"""
import markdown, pathlib

d = pathlib.Path(__file__).parent
md = markdown.Markdown(extensions=["toc", "tables", "fenced_code", "sane_lists"], extension_configs={"toc": {"toc_depth": "2-3"}})
body = md.convert((d / "MTMC.md").read_text())

CSS = """
:root { --bg:#fff; --fg:#1f2328; --mut:#656d76; --bd:#d0d7de; --code:#f6f8fa; --acc:#0969da; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --mut:#8b949e; --bd:#30363d; --code:#161b22; --acc:#58a6ff; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 -apple-system,Segoe UI,Helvetica,Arial,sans-serif; }
.layout { display:flex; max-width:1400px; margin:0 auto; }
nav { position:sticky; top:0; align-self:flex-start; width:300px; max-height:100vh; overflow:auto; padding:24px 16px; border-right:1px solid var(--bd); font-size:14px; }
nav .toc ul { list-style:none; margin:0; padding-left:14px; }
nav .toc > ul { padding-left:0; }
nav a { color:var(--fg); text-decoration:none; display:block; padding:2px 0; }
nav a:hover { color:var(--acc); }
main { flex:1; min-width:0; padding:24px 48px 80px; }
h1,h2,h3 { line-height:1.25; }
h1 { border-bottom:1px solid var(--bd); padding-bottom:.3em; }
h2 { border-bottom:1px solid var(--bd); padding-bottom:.3em; margin-top:2em; }
a { color:var(--acc); }
code { background:var(--code); padding:.15em .35em; border-radius:4px; font:85% ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
pre { background:var(--code); padding:14px 16px; border-radius:6px; overflow:auto; border:1px solid var(--bd); }
pre code { background:none; padding:0; font-size:85%; }
table { border-collapse:collapse; display:block; overflow:auto; margin:1em 0; }
th,td { border:1px solid var(--bd); padding:6px 12px; text-align:left; vertical-align:top; }
th { background:var(--code); }
hr { border:0; border-top:1px solid var(--bd); margin:2em 0; }
.toc-inline { display:none; }
@media (max-width:900px) { nav { display:none; } main { padding:16px; } }
@media print { nav { display:none; } main { padding:0; } }
"""

# the [TOC] marker in the Markdown becomes the sidebar; remove it from the body
toc = md.toc
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MTMC - Multi-Target Multi-Camera Tracking</title>
<style>{CSS}</style></head>
<body><div class="layout">
<nav>{toc}</nav>
<main>{body.replace(toc, "")}</main>
</div></body></html>
"""
(d / "MTMC.html").write_text(html)
print("wrote", d / "MTMC.html", len(html), "bytes")
