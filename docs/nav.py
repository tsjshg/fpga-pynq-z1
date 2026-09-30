#!/usr/bin/env python3
"""開発記録の各回に、目次・前後・言語切り替えのナビを入れる（何度走らせてもよい）。

    python3 docs/nav.py

回を足したら ENTRIES に 1 行足して走らせる。ja/ と en/ の全ページのナビを作り直す
（前の回の「次へ」も付く）。ナビは <!-- nav --> … <!-- /nav --> の間だけを書き換えるので、
本文には触らない。
"""
import os, re

D = os.path.dirname(os.path.abspath(__file__))
ENTRIES = [  # (ファイル名, 日本語の題, 英語の題)
    ("01-cpu-to-fabric.html", "CPU からファブリックへ", "From CPU to Fabric"),
    ("02-two-ceilings.html", "二つの天井", "Two Ceilings"),
    ("03-the-wall-does-not-move.html", "壁は動かない", "The Wall Doesn't Move"),
    ("04-reaching-the-wall.html", "壁に届いた", "Reaching the Wall"),
    ("05-end-to-end.html", "通しで動いた", "It Runs End to End"),
]
LABEL = {
    "ja": dict(index="開発記録の目次", prev="前", next="次", other="en", other_label="English",
               aria="開発記録のナビゲーション"),
    "en": dict(index="Journal index", prev="Previous", next="Next", other="ja", other_label="日本語",
               aria="Journal navigation"),
}
STYLE = """<!-- nav -->
<style>
  body{margin:0}
  img{max-width:100%}
  .jnav{max-width:760px;margin:0 auto;padding:14px 16px 0;display:flex;flex-wrap:wrap;
        gap:4px 18px;font-size:12.5px;line-height:1.7;opacity:.8}
  .jnav.foot{padding:0 16px 40px}
  .jnav a{color:inherit;text-underline-offset:3px;text-decoration-thickness:1px}
  .jnav a:focus-visible{outline:2px solid currentColor;outline-offset:2px}
  .jnav .lang{margin-left:auto}
  @media (min-width:600px){ .jnav{padding:14px 24px 0} .jnav.foot{padding:0 24px 40px} }
</style>
<!-- /nav -->
"""
BLOCK = re.compile(r"<!-- nav -->.*?<!-- /nav -->\n?", re.S)


def nav(lang, i, foot=False):
    t = LABEL[lang]
    title = (lambda k: ENTRIES[k][1]) if lang == "ja" else (lambda k: ENTRIES[k][2])
    a = [f'<a href="../index.html">{t["index"]}</a>']
    if i > 0:
        a.append(f'<a href="{ENTRIES[i-1][0]}" rel="prev">← {t["prev"]}: {title(i-1)}</a>')
    if i + 1 < len(ENTRIES):
        a.append(f'<a href="{ENTRIES[i+1][0]}" rel="next">{t["next"]}: {title(i+1)} →</a>')
    a.append(f'<a class="lang" href="../{t["other"]}/{ENTRIES[i][0]}" hreflang="{t["other"]}" '
             f'lang="{t["other"]}">{t["other_label"]}</a>')
    cls = "jnav foot" if foot else "jnav"
    return (f'<!-- nav -->\n<nav class="{cls}" aria-label="{t["aria"]}">\n  '
            + "\n  ".join(a) + "\n</nav>\n<!-- /nav -->\n")


def main():
    for lang in ("ja", "en"):
        for i, (name, _, _) in enumerate(ENTRIES):
            p = os.path.join(D, lang, name)
            if not os.path.exists(p):
                print(f"missing {lang}/{name}"); continue
            s = BLOCK.sub("", open(p, encoding="utf-8").read())
            s = re.sub(r'<html lang="[a-z]+">', f'<html lang="{lang}">', s, count=1)
            s = s.replace("</head>", STYLE + "</head>", 1)
            s = re.sub(r"<body>\n?", "<body>\n" + nav(lang, i), s, count=1)
            s = re.sub(r"\n?</body>", "\n" + nav(lang, i, foot=True) + "</body>", s, count=1)
            open(p, "w", encoding="utf-8").write(s)
            print(f"{lang}/{name}")


if __name__ == "__main__":
    main()
