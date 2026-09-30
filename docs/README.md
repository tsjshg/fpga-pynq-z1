# Development journal / 開発記録

How a ternary LLM ended up running on hand-written circuits in a PYNQ-Z1's FPGA fabric.
One entry per stretch of work, written at the time. Wrong turns and later corrections are left in.

PYNQ-Z1 の FPGA に自分で書いた回路で三値の言語モデルを動かすまでの記録です。
区切りごとにその時点で書いたもので、途中の誤りと後からの訂正もそのまま残してあります。

**Read it online / Web で読む: <https://tsjshg.github.io/fpga-pynq-z1/>**

Each entry is a self-contained HTML page, and GitHub shows `.html` files as source. Use the site
above, or clone the repository and open [`index.html`](index.html).
各回は 1 枚で完結した HTML です。GitHub 上ではソースとして表示されるので、上のサイトで読むか、
クローンして [`index.html`](index.html) を開いてください。

| # | Date | Stages | English | 日本語 | Later corrections / 後の訂正 |
|---|---|---|---|---|---|
| 01 | 2026-09-05 | 0–1 | [From CPU to Fabric](en/01-cpu-to-fabric.html) | [CPU からファブリックへ](ja/01-cpu-to-fabric.html) | "74% of DDR" was the datapath width, not DDR (→ 02) |
| 02 | 2026-09-08 | 2 | [Two Ceilings](en/02-two-ceilings.html) | [二つの天井](ja/02-two-ceilings.html) | 1.97 GB/s → 2.01 GB/s after removing measurement overhead (→ 04) |
| 03 | 2026-09-08 | 2–4 | [The Wall Doesn't Move](en/03-the-wall-does-not-move.html) | [壁は動かない](ja/03-the-wall-does-not-move.html) | Bandwidth 1–4% low (→ 04); CPU baseline built without NEON, ~1.8× too slow (→ 04, 05) |
| 04 | 2026-09-18 | 5a–5b | [Reaching the Wall](en/04-reaching-the-wall.html) | [壁に届いた](ja/04-reaching-the-wall.html) | |
| 05 | 2026-09-28 | 6–11 | [It Runs End to End](en/05-end-to-end.html) | [通しで動いた](ja/05-end-to-end.html) | |

New to FPGAs? Start with 01. Just want the result? Read 05.
FPGA が初めてなら 01 から。いまの到達点だけなら 05 を。

## Adding an entry / 回を足すとき

1. Write the page in Japanese as `ja/NN-slug.html` and translate it to `en/NN-slug.html`. Use the
   next number and the same file name in both languages. Start from a copy of the previous
   entry so it shares the palette and fonts. The page must be self-contained, with inline SVG
   and no scripts.
2. Link to earlier entries with relative file names (`04-reaching-the-wall.html`). Don't edit
   old entries except to add a clearly dated correction note.
3. Add the entry to `ENTRIES` in [`nav.py`](nav.py) and run `python3 docs/nav.py`. It rebuilds the
   navigation bars on every page: index, previous, next and the other language.
4. Add the entry to [`index.html`](index.html) and to the table above.
