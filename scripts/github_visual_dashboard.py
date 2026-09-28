from __future__ import annotations

import re
from pathlib import Path

MARKER = 'data-luxury-theme="v1"'


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "site" / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    document = index_path.read_text(encoding="utf-8")
    if MARKER in document:
        return 0

    document = document.replace("</style>", _styles() + "\n</style>", 1)
    document = re.sub(
        r"<body([^>]*)>",
        lambda match: f'<body{match.group(1)}>\n{_background()}',
        document,
        count=1,
    )
    index_path.write_text(document, encoding="utf-8")
    return 0


def _background() -> str:
    return (
        f'<div class="luxury-backdrop" {MARKER} aria-hidden="true">'
        '<span class="luxury-glow glow-one"></span>'
        '<span class="luxury-glow glow-two"></span>'
        "</div>"
    )


def _styles() -> str:
    return r"""
:root{
  color-scheme:light;
  --bg:#f2eadf;
  --paper:rgba(255,252,246,.90);
  --ink:#352820;
  --muted:#806e60;
  --line:rgba(91,64,43,.15);
  --sage:#b89243;
  --sage2:#6f4d25;
  --mint:#f1dfb6;
  --lav:#c8a65d;
  --lav2:#f4ead2;
  --peach:#a76f4d;
  --peach2:#efe0d2;
  --ok:#756f49;
  --bad:#9b5d50;
  --wait:#a68035;
  --shadow:0 18px 48px rgba(74,51,34,.09);
  --surface-strong:rgba(255,252,246,.96);
  --surface-soft:rgba(251,245,236,.88);
  --surface-faint:rgba(247,238,226,.74);
}
html{background:var(--bg)}
body{
  position:relative;
  isolation:isolate;
  overflow-x:hidden;
  background:
    linear-gradient(180deg,rgba(255,252,246,.94),rgba(242,234,223,.96)),
    var(--bg);
}
.luxury-backdrop{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.luxury-glow{position:absolute;border-radius:50%;filter:blur(18px);opacity:.22}
.glow-one{width:420px;height:420px;right:-150px;top:-180px;background:#d6b66f}
.glow-two{width:360px;height:360px;left:-170px;bottom:-170px;background:#c89570}
.shell{position:relative;z-index:1}
.top{margin-bottom:18px}
.logo{
  border-radius:12px;
  background:linear-gradient(145deg,#f5e8c7,#dfc17b);
  color:#68471f;
  box-shadow:inset 0 0 0 1px rgba(111,77,37,.10)
}
.brand h1{font-weight:760;letter-spacing:-.015em}
.brand p{color:var(--muted)}
.health,.health-summary,.health-badge{
  background:rgba(255,252,246,.78);
  border-color:var(--line);
  box-shadow:none
}
.hero{
  border-radius:24px;
  border-color:rgba(117,83,51,.13);
  background:linear-gradient(145deg,rgba(255,253,248,.97),rgba(246,237,224,.90));
  box-shadow:var(--shadow)
}
.eyebrow,.structure-eyebrow{color:#8b682c}
.direction.up{color:#866328}
.direction.down{color:#8f5548}
.confidence{
  border:1px solid rgba(181,138,53,.18);
  background:#f4e7c8;
  color:#71501f
}
.track{background:#eadbc8}
.fill{background:linear-gradient(90deg,#76502c,#c59b43)}
.forecast-card,.metric,.panel,.learn,.scroll,.structure-tile,.trade-lifecycle-tile,.boundary-memory-tile{
  background:var(--paper);
  border-color:var(--line);
  box-shadow:0 10px 28px rgba(74,51,34,.055);
  backdrop-filter:none
}
.metric{border-radius:17px}
.panel{border-radius:20px}
.forecast-card{border-radius:18px}
.range{background:linear-gradient(90deg,#efe0d2,#f3e7ca,#ead6a8)}
.range-values{color:#755f50}
.line{stroke:#79572e}
.band{fill:rgba(190,151,73,.14)}
.median{stroke:#b1893d}
.correct{fill:#eee8d4;stroke:#756f49}
.wrong{fill:#f1dfd8;stroke:#9b5d50}
.pending-dot{fill:#f1e6c9;stroke:#a68035}
.mini{background:#eadfd1}
.mini div{background:linear-gradient(90deg,#76502c,#bd9342)}
.chip{
  background:#f1e3d4;
  color:#785440;
  border-color:rgba(145,92,62,.14)
}
th{background:rgba(249,242,232,.98)}
.pill.up{background:#f0e2bd;color:#74521f}
.pill.down{background:#efddd4;color:#8a5046}
.result-ok{color:#625f40;background:rgba(117,111,73,.10)}
.result-bad{color:#8d5148;background:rgba(155,93,80,.10)}
.result-wait{color:#8b6a2e;background:rgba(166,128,53,.10)}
footer a{color:#76502c}
.structure-panel,.economic-panel,.boundary-memory-panel,.trade-lifecycle-panel{
  background:linear-gradient(145deg,rgba(255,252,246,.94),rgba(243,232,215,.82))
}
.structure-action,.economic-action{
  background:#f1e1ba;
  color:#714f1f
}
@media(max-width:620px){
  .hero{border-radius:20px}
  .panel{border-radius:18px}
  .luxury-glow{opacity:.14}
}
@media(prefers-reduced-motion:reduce){
  *{scroll-behavior:auto!important}
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
