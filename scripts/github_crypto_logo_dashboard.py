from __future__ import annotations

import re
import shutil
from pathlib import Path

MARKER = 'data-crypto-logo-strip="v1"'
COINS = (
    ("btc", "Bitcoin"),
    ("eth", "Ethereum"),
    ("usdt", "Tether"),
    ("bnb", "BNB"),
    ("sol", "Solana"),
    ("xrp", "XRP"),
    ("ada", "Cardano"),
    ("doge", "Dogecoin"),
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    index_path = site_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    _copy_assets(root, site_dir)

    document = index_path.read_text(encoding="utf-8")
    if MARKER in document:
        return 0

    document = document.replace("</style>", _styles() + "\n</style>", 1)
    document, replacements = re.subn(
        r'<nav class="top">.*?</nav>',
        _strip(),
        document,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Dashboard header could not be replaced")

    index_path.write_text(document, encoding="utf-8")
    return 0


def _copy_assets(root: Path, site_dir: Path) -> None:
    source_dir = root / "assets" / "crypto"
    target_dir = site_dir / "assets" / "crypto"
    target_dir.mkdir(parents=True, exist_ok=True)
    for symbol, _ in COINS:
        source = source_dir / f"{symbol}.svg"
        if not source.exists():
            raise FileNotFoundError(f"Missing crypto logo: {source}")
        shutil.copyfile(source, target_dir / source.name)


def _strip() -> str:
    nodes = "".join(
        (
            '<span class="crypto-logo-node">'
            f'<span class="crypto-logo-medallion">'
            f'<img src="assets/crypto/{symbol}.svg" alt="{name}" '
            f'title="{name}" width="32" height="32">'
            "</span>"
            "</span>"
        )
        for symbol, name in COINS
    )
    return (
        f'<nav class="crypto-logo-strip" {MARKER} '
        'aria-label="Cryptocurrency symbols">'
        f'<div class="crypto-logo-chain">{nodes}</div>'
        "</nav>"
    )


def _styles() -> str:
    return r"""
.crypto-logo-strip{
  margin:0 0 18px;
  padding:14px 16px;
  border:1px solid rgba(111,77,37,.13);
  border-radius:20px;
  background:linear-gradient(145deg,rgba(255,253,248,.94),rgba(244,234,213,.84));
  box-shadow:0 10px 30px rgba(74,51,34,.055);
}
.crypto-logo-chain{
  display:flex;
  align-items:center;
  justify-content:center;
  min-width:max-content;
  overflow-x:auto;
  padding:2px 3px;
  scrollbar-width:none;
  -ms-overflow-style:none;
}
.crypto-logo-chain::-webkit-scrollbar{display:none}
.crypto-logo-node{
  display:flex;
  align-items:center;
  flex:0 0 auto;
}
.crypto-logo-node:not(:last-child)::after{
  content:"";
  width:clamp(16px,3.2vw,44px);
  height:1px;
  margin:0 7px;
  background:linear-gradient(90deg,rgba(118,80,44,.16),rgba(197,155,67,.72),rgba(118,80,44,.16));
}
.crypto-logo-medallion{
  width:54px;
  height:54px;
  display:grid;
  place-items:center;
  border-radius:50%;
  border:1px solid rgba(181,138,53,.22);
  background:linear-gradient(145deg,#fffdf8,#f2e3c5);
  box-shadow:
    0 7px 18px rgba(74,51,34,.08),
    inset 0 0 0 4px rgba(255,255,255,.42);
}
.crypto-logo-medallion img{
  width:38px;
  height:38px;
  display:block;
}
@media(max-width:720px){
  .crypto-logo-strip{
    margin-bottom:14px;
    padding:11px 10px;
    border-radius:17px;
    overflow:hidden;
  }
  .crypto-logo-chain{
    justify-content:flex-start;
  }
  .crypto-logo-medallion{
    width:46px;
    height:46px;
  }
  .crypto-logo-medallion img{
    width:34px;
    height:34px;
  }
  .crypto-logo-node:not(:last-child)::after{
    width:18px;
    margin:0 5px;
  }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
