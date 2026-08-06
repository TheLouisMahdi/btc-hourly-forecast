from __future__ import annotations

import re
from pathlib import Path

MARKER = 'data-market-ambience="v1"'


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "site" / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    document = index_path.read_text(encoding="utf-8")
    if MARKER in document:
        return 0

    document = _inject_early_theme(document)
    document = document.replace("</style>", _styles() + "\n</style>", 1)
    document = re.sub(
        r"<body([^>]*)>",
        lambda match: f'<body{match.group(1)}>\n{_background()}',
        document,
        count=1,
    )
    document = _inject_theme_control(document)
    document = document.replace("</body>", _script() + "\n</body>", 1)
    index_path.write_text(document, encoding="utf-8")
    return 0


def _inject_early_theme(document: str) -> str:
    script = r'''<script data-market-ambience="theme-bootstrap">
(function(){
  try {
    var saved = localStorage.getItem("btc-dashboard-theme");
    var theme = saved || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  } catch (_) {
    document.documentElement.setAttribute("data-theme", "light");
  }
})();
</script>'''
    return document.replace("<head>", "<head>\n" + script, 1)


def _inject_theme_control(document: str) -> str:
    control = r'''
<button class="theme-toggle" id="themeToggle" type="button" aria-label="Switch color theme" aria-pressed="false">
  <span class="theme-toggle-icon" aria-hidden="true">☾</span>
  <span class="theme-toggle-copy"><small>APPEARANCE</small><strong>Dark</strong></span>
</button>'''
    pattern = re.compile(r'(<div class="health-stack">.*?</div>)', re.DOTALL)
    if pattern.search(document):
        return pattern.sub(
            lambda match: '<div class="top-controls">' + match.group(1) + control + "</div>",
            document,
            count=1,
        )
    return document.replace("</nav>", control + "\n</nav>", 1)


def _background() -> str:
    coins = [
        ("btc", "₿", "7%", "11%", "88px", "34s", "-11s", "92px", "118px", "-62px", "74px", "38px", "-48px"),
        ("eth", "Ξ", "80%", "13%", "74px", "41s", "-27s", "-108px", "94px", "54px", "-72px", "-36px", "44px"),
        ("sol", "S", "17%", "68%", "68px", "37s", "-18s", "126px", "-66px", "44px", "88px", "-52px", "-36px"),
        ("xrp", "X", "72%", "72%", "62px", "46s", "-7s", "-94px", "-102px", "72px", "38px", "34px", "82px"),
        ("bnb", "B", "45%", "8%", "58px", "39s", "-30s", "72px", "86px", "-84px", "48px", "38px", "-62px"),
        ("ada", "A", "91%", "45%", "66px", "44s", "-22s", "-86px", "48px", "-42px", "-92px", "54px", "78px"),
        ("doge", "Ð", "35%", "84%", "72px", "49s", "-15s", "104px", "-88px", "-76px", "-48px", "64px", "52px"),
    ]
    items = []
    for name, symbol, left, top, size, duration, delay, dx1, dy1, dx2, dy2, dx3, dy3 in coins:
        items.append(
            f'<span class="ambient-coin coin-{name}" style="--left:{left};--top:{top};--size:{size};'
            f'--duration:{duration};--delay:{delay};--dx1:{dx1};--dy1:{dy1};'
            f'--dx2:{dx2};--dy2:{dy2};--dx3:{dx3};--dy3:{dy3}">'
            f'<span>{symbol}</span></span>'
        )
    return (
        f'<div class="ambient-market" {MARKER} aria-hidden="true">'
        + "".join(items)
        + '<span class="ambient-heart heart-one">♥</span>'
        + '<span class="ambient-heart heart-two">♥</span>'
        + "</div>"
    )


def _styles() -> str:
    return r'''
:root{
  --surface-strong:rgba(255,255,255,.92);
  --surface-soft:rgba(255,255,255,.60);
  --surface-faint:rgba(255,255,255,.46);
  --coin-opacity:.095;
  --coin-filter:saturate(.82) brightness(1.03);
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0b1212;
  --paper:rgba(17,29,27,.82);
  --ink:#eef6f3;
  --muted:#91a49f;
  --line:rgba(178,218,207,.15);
  --sage:#82b8ab;
  --sage2:#9bd0c3;
  --mint:rgba(69,116,105,.25);
  --lav:#b2a9dd;
  --lav2:rgba(111,99,154,.24);
  --peach:#d7a08d;
  --peach2:rgba(150,91,72,.23);
  --ok:#77b89f;
  --bad:#de8e89;
  --wait:#d1b875;
  --shadow:0 24px 80px rgba(0,0,0,.34);
  --surface-strong:rgba(20,34,32,.94);
  --surface-soft:rgba(24,39,36,.74);
  --surface-faint:rgba(26,43,40,.58);
  --coin-opacity:.12;
  --coin-filter:saturate(.92) brightness(1.08);
}
html{background:var(--bg);transition:background-color .25s ease,color .25s ease}
body{position:relative;isolation:isolate;overflow-x:hidden;transition:background .35s ease,color .25s ease}
:root[data-theme="dark"] body{
  background:
    radial-gradient(circle at 12% 4%,rgba(72,120,108,.24),transparent 31%),
    radial-gradient(circle at 90% 3%,rgba(105,94,148,.22),transparent 29%),
    radial-gradient(circle at 52% 94%,rgba(143,89,70,.13),transparent 34%),
    linear-gradient(180deg,#0d1716,var(--bg));
}
.shell{position:relative;z-index:2}
.top{position:relative;z-index:4;flex-wrap:wrap}
.top-controls{display:flex;align-items:center;justify-content:flex-end;gap:10px;margin-left:auto}
.theme-toggle{appearance:none;display:flex;align-items:center;gap:9px;min-height:42px;padding:7px 11px 7px 8px;border:1px solid var(--line);border-radius:15px;background:var(--surface-strong);color:var(--ink);box-shadow:0 10px 30px rgba(55,80,73,.07);cursor:pointer;font:inherit;transition:transform .2s ease,background .25s ease,border-color .25s ease}
.theme-toggle:hover{transform:translateY(-1px)}
.theme-toggle:focus-visible{outline:3px solid rgba(111,155,145,.28);outline-offset:2px}
.theme-toggle-icon{width:28px;height:28px;display:grid;place-items:center;border-radius:10px;background:linear-gradient(135deg,var(--mint),var(--lav2),var(--peach2));color:var(--sage2);font-size:16px;font-weight:900}
.theme-toggle-copy{display:grid;text-align:left;line-height:1.05}.theme-toggle-copy small{font-size:7px;letter-spacing:.09em;color:var(--muted)}.theme-toggle-copy strong{margin-top:3px;font-size:10px}
.ambient-market{position:fixed;inset:-8vh -7vw;z-index:0;overflow:hidden;pointer-events:none;contain:strict}
.ambient-market:after{content:"";position:absolute;inset:0;background:radial-gradient(circle at center,transparent 25%,rgba(243,247,245,.28) 100%);transition:background .3s ease}
:root[data-theme="dark"] .ambient-market:after{background:radial-gradient(circle at center,transparent 22%,rgba(5,10,10,.42) 100%)}
.ambient-coin{--coin-color:#d9a86c;position:absolute;left:var(--left);top:var(--top);width:var(--size);height:var(--size);display:grid;place-items:center;border-radius:50%;opacity:var(--coin-opacity);filter:var(--coin-filter);animation:coin-drift var(--duration) cubic-bezier(.45,.05,.55,.95) var(--delay) infinite alternate,coin-light calc(var(--duration)*.31) ease-in-out var(--delay) infinite;will-change:transform,opacity,filter}
.ambient-coin:before,.ambient-coin:after{content:"";position:absolute;border-radius:inherit}.ambient-coin:before{inset:0;border:1px solid color-mix(in srgb,var(--coin-color) 55%,transparent);background:radial-gradient(circle at 31% 24%,color-mix(in srgb,var(--coin-color) 52%,white),color-mix(in srgb,var(--coin-color) 25%,transparent) 38%,color-mix(in srgb,var(--coin-color) 12%,transparent) 72%);box-shadow:inset -9px -12px 20px rgba(0,0,0,.08),0 0 26px color-mix(in srgb,var(--coin-color) 22%,transparent)}
.ambient-coin:after{inset:12%;border:1px solid color-mix(in srgb,var(--coin-color) 42%,transparent)}
.ambient-coin>span{position:relative;z-index:1;color:color-mix(in srgb,var(--coin-color) 78%,var(--ink));font-size:calc(var(--size)*.42);font-weight:900;text-shadow:0 0 18px color-mix(in srgb,var(--coin-color) 50%,transparent);transform:rotate(-8deg)}
.coin-btc{--coin-color:#f2a84b}.coin-eth{--coin-color:#9a91d1}.coin-sol{--coin-color:#65c7aa}.coin-xrp{--coin-color:#8ea7aa}.coin-bnb{--coin-color:#e3bd59}.coin-ada{--coin-color:#6fa9d8}.coin-doge{--coin-color:#cba86a}
.ambient-heart{position:absolute;color:var(--peach);opacity:.055;font-size:56px;filter:blur(.2px);animation:heart-float 24s ease-in-out infinite alternate}.heart-one{left:57%;top:17%;animation-delay:-8s}.heart-two{left:9%;top:49%;font-size:38px;color:var(--lav);animation-delay:-17s}
@keyframes coin-drift{0%{transform:translate3d(0,0,0) rotate(-8deg) scale(.96)}28%{transform:translate3d(var(--dx1),var(--dy1),0) rotate(72deg) scale(1.03)}57%{transform:translate3d(var(--dx2),var(--dy2),0) rotate(166deg) scale(.94)}79%{transform:translate3d(var(--dx3),var(--dy3),0) rotate(248deg) scale(1.02)}100%{transform:translate3d(calc(var(--dx1)*-.35),calc(var(--dy2)*-.35),0) rotate(350deg) scale(.98)}}
@keyframes coin-light{0%,100%{opacity:calc(var(--coin-opacity)*.66);filter:var(--coin-filter) brightness(.88)}48%{opacity:calc(var(--coin-opacity)*1.18);filter:var(--coin-filter) brightness(1.22)}72%{opacity:calc(var(--coin-opacity)*.82);filter:var(--coin-filter) brightness(1.02)}}
@keyframes heart-float{0%{transform:translate3d(0,0,0) rotate(-12deg)}50%{transform:translate3d(48px,-38px,0) rotate(9deg)}100%{transform:translate3d(-28px,42px,0) rotate(-4deg)}}
:root[data-theme="dark"] .hero{background:linear-gradient(135deg,rgba(22,37,34,.96),rgba(14,25,24,.82));border-color:rgba(177,218,207,.12)}
:root[data-theme="dark"] .metric,:root[data-theme="dark"] .panel{border-color:rgba(177,218,207,.12);box-shadow:0 16px 48px rgba(0,0,0,.22)}
:root[data-theme="dark"] .forecast-card,
:root[data-theme="dark"] .health-badge,
:root[data-theme="dark"] .structure-tile,
:root[data-theme="dark"] .trade-lifecycle-tile,
:root[data-theme="dark"] .boundary-memory-tile,
:root[data-theme="dark"] .learn,
:root[data-theme="dark"] .scroll,
:root[data-theme="dark"] .trade-route,
:root[data-theme="dark"] .position-summary-chip,
:root[data-theme="dark"] .exact-candle-timing div{background:var(--surface-soft);border-color:var(--line)}
:root[data-theme="dark"] .structure-panel,:root[data-theme="dark"] .economic-panel,:root[data-theme="dark"] .boundary-memory-panel,:root[data-theme="dark"] .trade-lifecycle-panel{background:linear-gradient(135deg,rgba(22,37,34,.91),rgba(47,41,63,.54),rgba(40,31,29,.34))}
:root[data-theme="dark"] th{background:rgba(18,31,29,.98)}
:root[data-theme="dark"] .range{background:linear-gradient(90deg,rgba(142,83,66,.25),rgba(105,94,148,.27),rgba(64,111,100,.28))}
:root[data-theme="dark"] .range-values{color:var(--muted)}
:root[data-theme="dark"] .track,:root[data-theme="dark"] .mini{background:rgba(255,255,255,.08)}
:root[data-theme="dark"] .chip{background:rgba(143,89,70,.20);color:#ddb9aa;border-color:rgba(215,160,141,.16)}
:root[data-theme="dark"] .axis{fill:var(--muted)}
:root[data-theme="dark"] .grid-line{stroke:rgba(178,218,207,.09)}
@media(max-width:760px){.top-controls{width:100%;justify-content:space-between}.health-stack{justify-content:flex-start}.ambient-coin{opacity:calc(var(--coin-opacity)*.78)}.ambient-coin:nth-child(n+6){display:none}}
@media(prefers-reduced-motion:reduce){.ambient-coin,.ambient-heart{animation:none!important}.theme-toggle{transition:none}}
'''


def _script() -> str:
    return r'''
<script data-market-ambience="controls">
(function(){
  var root = document.documentElement;
  var toggle = document.getElementById("themeToggle");
  var themeMeta = document.querySelector('meta[name="theme-color"]');

  function apply(theme, persist) {
    var normalized = theme === "dark" ? "dark" : "light";
    root.setAttribute("data-theme", normalized);
    if (persist) {
      try { localStorage.setItem("btc-dashboard-theme", normalized); } catch (_) {}
    }
    if (themeMeta) themeMeta.setAttribute("content", normalized === "dark" ? "#0b1212" : "#f3f7f5");
    if (!toggle) return;
    var icon = toggle.querySelector(".theme-toggle-icon");
    var label = toggle.querySelector(".theme-toggle-copy strong");
    var nextIsDark = normalized !== "dark";
    toggle.setAttribute("aria-pressed", normalized === "dark" ? "true" : "false");
    toggle.setAttribute("aria-label", nextIsDark ? "Switch to dark mode" : "Switch to light mode");
    if (icon) icon.textContent = normalized === "dark" ? "☀" : "☾";
    if (label) label.textContent = normalized === "dark" ? "Light" : "Dark";
  }

  apply(root.getAttribute("data-theme") || "light", false);
  if (toggle) {
    toggle.addEventListener("click", function(){
      apply(root.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
    });
  }
})();
</script>'''


if __name__ == "__main__":
    raise SystemExit(main())
