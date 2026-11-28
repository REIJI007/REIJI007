#!/usr/bin/env python3
"""
REIJI007 主页文字渲染器
-----------------------
GitHub README 会过滤所有 CSS / font 标签，无法指定字体。
因此把所有文字用「霞鹜文楷 LXGW WenKai」排成 SVG：
  - 每个 SVG 只嵌入实际用到的字形（WOFF2 子集，base64 内联），体积约几十 KB
  - 全部文字拼成一张自带底色的卡片，输出 light / dark 两版；
    README 用 <picture> + prefers-color-scheme 按访客自己的 GitHub 主题选择 light / dark
    （昼夜意象随主题固定：亮色版是阳伞，暗色版是点亮的烛台）
  - stats 卡片通过 GitHub GraphQL 实时取数（Actions 中运行，每日更新）

用法:
  python scripts/render.py            # 渲染全部（需要环境变量 GITHUB_TOKEN 才会更新 stats）
  python scripts/render.py --mock     # 用假数据渲染 stats，本地预览用
"""
import base64, io, json, math, os, sys, datetime, urllib.request
from zoneinfo import ZoneInfo
from pathlib import Path
from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "assets" / "fonts"
OUT = ROOT / "assets" / "svg"
USER = os.environ.get("PROFILE_USER", "REIJI007")
W = 830  # README 内容区宽度，SVG 以 width=100% 自适应缩放
TODAY = datetime.datetime.now(ZoneInfo("Asia/Taipei")).date()

# ───────────────────────── 配色（取自顶部插画）─────────────────────────
# 米白羊皮纸 / 深棕墨 / 酒红玫瑰 / 古金雕花 / 深蓝裙装
THEMES = {
    "light": dict(
        ink="#3b2a22", sub="#7a6455", accent="#8d3c42", gold="#a8844f",
        deep="#5a1b20", petal="#d9a3a6", candle="#f3ead9", bg="#fbf7f0", suits=".115", window=".26", lit=False, mote="#b8862e", sheen="#fff4d6", suit_glow="#d9a441",
        heat=["#efe6da", "#e3c6ab", "#cf957e", "#a8574f", "#7a2e33"],
    ),
    "dark": dict(
        ink="#ecdfcd", sub="#b3a08d", accent="#d08a8e", gold="#c9a46a",
        deep="#7a2e33", petal="#e8b9bb", candle="#efe3cf", bg="#15110f", suits=".105", window=".22", lit=True, mote="#f2b866", sheen="#e39a3c", suit_glow="#ffd27a",
        heat=["#2a221e", "#4a3229", "#7a3f38", "#b0625a", "#e0a0a0"],
    ),
}

# ───────────────────────── 动画（SVG 内联 CSS，README 中自动循环播放）─────────────────────────
# 原则：底纹缓慢漂移，双层环形流光由中心向外扩散；同时兼容减少动态效果设置。
ANIM_CSS = (
    # 扑克牌四花色底纹：缓慢漂移；两道环形流光从卡片中心向四周扩散
    ".suits-drift{transform-box:fill-box;transform-origin:center;animation:suitsDrift 48s ease-in-out infinite alternate}"
    "@keyframes suitsDrift{from{transform:translate(0,0) scale(1)}to{transform:translate(-30px,22px) scale(1.025)}}"
    ".suit-wave{fill:none;stroke-width:10;transform-box:fill-box;transform-origin:center;"
    "animation:suitWave 14s cubic-bezier(.15,.55,.35,1) infinite;opacity:0;"
    "filter:drop-shadow(0 0 8px rgba(246,201,119,.82))}"
    ".suit-wave.wave-b{animation-delay:-7s;stroke-width:8}"
    "@keyframes suitWave{0%{transform:scale(.015);opacity:0}10%{opacity:.88}"
    "78%{transform:scale(2.18);opacity:.24}100%{transform:scale(2.3);opacity:0}}"
    # 烛火：以火苗底部为支点明显地摇曳、拉长、压扁，亮度跳动
    ".flame{transform-box:fill-box;transform-origin:50% 100%;animation:flicker 1.6s ease-in-out infinite}"
    "@keyframes flicker{0%,100%{transform:scale(1,1) skewX(0);opacity:1}"
    "20%{transform:scale(.72,1.32) skewX(8deg);opacity:.75}"
    "45%{transform:scale(1.12,.86) skewX(-6deg);opacity:1}"
    "70%{transform:scale(.82,1.2) skewX(5deg);opacity:.82}}"
    # 烛光光晕：明显的明暗呼吸 + 缩放
    ".glow{transform-box:fill-box;transform-origin:50% 50%;animation:breathe 2.4s ease-in-out infinite}"
    "@keyframes breathe{0%,100%{opacity:.3;transform:scale(.82)}50%{opacity:1;transform:scale(1.18)}}"
    # 小鸟：每 4 秒左右歪头，再连跳两下
    ".bird{transform-box:fill-box;transform-origin:50% 100%;animation:hop 4s ease-in-out infinite}"
    "@keyframes hop{0%,45%,100%{transform:translateY(0) rotate(0)}"
    "52%{transform:translateY(0) rotate(-22deg)}60%{transform:translateY(0) rotate(16deg)}"
    "66%{transform:translateY(-6px) rotate(0)}72%{transform:translateY(0)}"
    "78%{transform:translateY(-5px) rotate(-6deg)}84%{transform:translateY(0) rotate(0)}}"
    # 氛围粒子：白天光尘斜飘，夜晚微光上升并闪烁
    ".mote{animation:var(--a) var(--d) ease-in-out var(--l) infinite;opacity:0}"
    "@keyframes dustA{0%{transform:translate(0,0);opacity:0}15%{opacity:var(--o)}85%{opacity:var(--o)}"
    "100%{transform:translate(130px,150px);opacity:0}}"
    "@keyframes dustB{0%{transform:translate(0,0);opacity:0}15%{opacity:var(--o)}85%{opacity:var(--o)}"
    "100%{transform:translate(-100px,160px);opacity:0}}"
    "@keyframes riseA{0%{transform:translate(0,0);opacity:0}15%{opacity:var(--o)}35%{opacity:.25}"
    "50%{transform:translate(24px,-90px);opacity:var(--o)}65%{opacity:.3}85%{opacity:var(--o)}"
    "100%{transform:translate(-12px,-180px);opacity:0}}"
    "@keyframes riseB{0%{transform:translate(0,0);opacity:0}15%{opacity:var(--o)}40%{opacity:.3}"
    "50%{transform:translate(-26px,-80px);opacity:var(--o)}70%{opacity:.25}85%{opacity:var(--o)}"
    "100%{transform:translate(14px,-170px);opacity:0}}"
    # 玫瑰窗：90 秒一圈缓慢旋转
    ".window{transform-box:fill-box;transform-origin:50% 50%;animation:spin 90s linear infinite}"
    "@keyframes spin{to{transform:rotate(360deg)}}"
    # 白天的蕾丝阳伞：以伞柄底端为支点轻轻左右摇摆
    ".parasol{transform-box:fill-box;transform-origin:50% 100%;animation:parasol 5s ease-in-out infinite}"
    "@keyframes parasol{0%,100%{transform:rotate(-5deg)}50%{transform:rotate(5deg)}}"
    # 茶杯热气：三缕细烟错开节奏，从杯口升起、左右轻摆、渐散
    ".steam{transform-box:fill-box;transform-origin:50% 100%;animation:steam 3.6s ease-in-out infinite;opacity:0}"
    "@keyframes steam{0%{transform:translateY(3px) skewX(0);opacity:0}30%{opacity:.9}"
    "55%{transform:translateY(-2px) skewX(10deg)}80%{opacity:.5}"
    "100%{transform:translateY(-7px) skewX(-8deg);opacity:0}}"
    # 季节花瓣：边旋转边摇摆着飘落（下落距离随卡片高度，见 petals()）
    ".petal{transform-box:fill-box;transform-origin:50% 50%;animation:var(--a) var(--d) linear var(--l) infinite;opacity:0}"
    # 热力图：活跃格子各自闪烁（越活跃越亮），外圈光晕起伏；最活跃的日子偶尔亮起四角星芒
    ".spark{animation:twinkle var(--d) ease-in-out var(--l) infinite;opacity:0}"
    "@keyframes twinkle{0%,100%{opacity:0}50%{opacity:var(--o)}}"
    ".halo{transform-box:fill-box;transform-origin:50% 50%;animation:halo var(--d) ease-in-out var(--l) infinite;opacity:0}"
    "@keyframes halo{0%,100%{opacity:0;transform:scale(.7)}50%{opacity:var(--o);transform:scale(1.3)}}"
    ".glint{transform-box:fill-box;transform-origin:50% 50%;animation:glint var(--d) ease-out var(--l) infinite;opacity:0}"
    "@keyframes glint{0%,78%,100%{opacity:0;transform:scale(0) rotate(0)}87%{opacity:1;transform:scale(1.35) rotate(45deg)}"
    "96%{opacity:0;transform:scale(.4) rotate(90deg)}}"
    # 标题字母：每 6 秒从左到右依次跳起，形成波浪
    ".ltr{transform-box:fill-box;transform-origin:50% 100%;animation:wave 6s ease-in-out infinite;"
    "animation-delay:calc(var(--i) * .1s);cursor:default}"
    "@keyframes wave{0%,22%,100%{transform:translateY(0) rotate(0)}"
    "6%{transform:translateY(-14px) rotate(-5deg)}12%{transform:translateY(3px) rotate(2deg)}17%{transform:translateY(0)}}"
    # 悬停：README 中图片不接收鼠标事件，仅在单独打开 SVG 时生效
    ".ltr:hover,.hop:hover{animation:pop .5s ease-out;transform:translateY(-6px) scale(1.12)}"
    ".hop{transform-box:fill-box;transform-origin:50% 100%;transition:transform .25s}"
    "@keyframes pop{0%{transform:translateY(0)}40%{transform:translateY(-16px) scale(1.18)}100%{transform:translateY(-6px) scale(1.12)}}"
    "@media (prefers-reduced-motion:reduce){.ltr{animation:none}}"
    "@media (prefers-reduced-motion:reduce){.suits-drift,.suit-wave{animation:none!important;filter:none}"
    ".flame,.glow,.bird,.mote,.window,.steam,.parasol,.petal,.spark,.halo,.glint{animation:none}"
    ".mote,.petal,.spark,.halo,.glint{opacity:0}}"
)

def motes(c, H):
    """
    昼夜氛围粒子（固定随机种子，保证每次渲染位置一致、不产生无意义的提交）
    白天：金色光尘，斜向飘落，像窗外阳光里的灰尘
    夜晚：暖色微光，一边上升一边闪烁，像烛光映出的微粒
    """
    import random
    rnd = random.Random(20261125)
    day = not c["lit"]
    out = ('<defs><radialGradient id="moteg"><stop offset="0" stop-color="#fff8e6"/>'
           f'<stop offset=".35" stop-color="{c["mote"]}"/>'
           f'<stop offset="1" stop-color="{c["mote"]}" stop-opacity="0"/></radialGradient></defs>')
    for i in range(26):
        x = rnd.uniform(50, W - 50)
        y = rnd.uniform(40, H - 40)
        r = rnd.uniform(5, 9.5)
        anim = rnd.choice(["dustA", "dustB"] if day else ["riseA", "riseB"])
        dur = rnd.uniform(8, 14)
        delay = -rnd.uniform(0, dur)            # 负延迟：打开页面时粒子已在不同阶段，不会同时出现
        op = rnd.uniform(.7, 1)
        out += (f'<circle class="mote" cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="url(#moteg)" '
                f'style="--a:{anim};--d:{dur:.1f}s;--l:{delay:.1f}s;--o:{op:.2f}"/>')
    return out

def petals(c, H):
    """季节花瓣飘落：冬 山茶花瓣 / 春 樱瓣 / 夏 蔷薇花瓣 / 秋 桂花（与季节分隔线同步）"""
    import random
    rnd = random.Random(1125)
    season = SEASON[TODAY.month]
    shapes = {
        "camellia": f'<path d="M0 -6 C5 -6 7 -1 5 3 C3 6 -3 6 -5 3 C-7 -1 -5 -6 0 -6 Z" fill="#a83a44" stroke="#5a1b20" stroke-width=".5"/>',
        "sakura": f'<path d="M0 6 C-5 2 -5 -4 -2 -6 L0 -4.6 L2 -6 C5 -4 5 2 0 6 Z" fill="{c["petal"]}" stroke="{c["accent"]}" stroke-width=".5"/>',
        "rose": '<path d="M0 -7 C6 -6 7 0 4 4 C2 6 -2 6 -4 4 C-7 0 -6 -6 0 -7 Z" fill="#7a2430" stroke="#4d161b" stroke-width=".5"/>',
        "osmanthus": "".join(f'<circle cx="{1.9*math.cos(math.radians(45+90*k)):.2f}" cy="{1.9*math.sin(math.radians(45+90*k)):.2f}" '
                             f'r="1.9" fill="#d9a441"/>' for k in range(4)) + '<circle r=".9" fill="#a8574f"/>',
    }
    shape = shapes[season]
    fall = lambda name, sw: (
        f"@keyframes {name}{{0%{{transform:translate(0,0) rotate(0);opacity:0}}6%{{opacity:.92}}"
        f"25%{{transform:translate({28*sw}px,{H*.25:.0f}px) rotate({100*sw}deg)}}"
        f"50%{{transform:translate({-16*sw}px,{H*.5:.0f}px) rotate({200*sw}deg)}}"
        f"75%{{transform:translate({24*sw}px,{H*.75:.0f}px) rotate({290*sw}deg)}}92%{{opacity:.92}}"
        f"100%{{transform:translate(0,{H+40:.0f}px) rotate({380*sw}deg);opacity:0}}}}")
    out = f"<style>{fall('fallA', 1)}{fall('fallB', -1)}</style>"
    for i in range(12):
        x = rnd.uniform(40, W - 40)
        sc = rnd.uniform(1, 1.7) * (1.4 if season == "osmanthus" else 1)
        dur = rnd.uniform(11, 18)
        delay = -rnd.uniform(0, dur)
        anim = rnd.choice(["fallA", "fallB"])
        out += (f'<g transform="translate({x:.0f} -20) scale({sc:.2f})"><g class="petal" '
                f'style="--a:{anim};--d:{dur:.1f}s;--l:{delay:.1f}s">{shape}</g></g>')
    return out

# ───────────────────────── 字体子集化 ─────────────────────────
_font_cache: dict = {}

def font_face(text: str, weight: str) -> str:
    """返回只包含 text 中字形的 @font-face 规则（WOFF2 base64 内联）"""
    key = (weight, "".join(sorted(set(text))))
    if key in _font_cache:
        return _font_cache[key]
    src = FONT_DIR / f"LXGWWenKai-{weight}.subset.woff2"
    font = TTFont(src, recalcTimestamp=False)   # 固定时间戳：输出可复现，内容没变就不会产生新提交
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "palt"]
    opts.name_IDs = ["*"]           # 保留版权与许可信息（OFL 要求）
    opts.notdef_outline = True
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    fam = f"WK{weight}"
    rule = f"@font-face{{font-family:'{fam}';src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
    _font_cache[key] = rule
    return rule

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def svg(height: int, body: str, texts: dict, title: str) -> str:
    """
    组装完整 SVG。texts = {weight: 该字重用到的全部字符串}
    CSS 中 .r / .m 分别对应 Regular / Medium 字重
    """
    faces = "".join(font_face(t, w) for w, t in texts.items() if t)
    css = faces + (
        ".r{font-family:'WKRegular',serif}"
        ".m{font-family:'WKMedium',serif}"
    ) + ANIM_CSS
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" role="img" aria-label="{esc(title)}">'
        f"<title>{esc(title)}</title><style>{css}</style>{body}</svg>"
    )

def text(x, y, s, size, fill, cls="r", anchor="middle", spacing=0, opacity=1):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    op = f' opacity="{opacity}"' if opacity != 1 else ""
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}"{ls}{op}>{esc(s)}</text>')

def ornament(cx, cy, c, half=150):
    """古典分隔饰纹：两端渐隐的细金线 + 中心菱形 + 两侧小圆点"""
    gid = f"g{cx}{cy}"
    return (
        f'<defs><linearGradient id="{gid}l" x1="0" x2="1"><stop offset="0" stop-color="{c["gold"]}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{c["gold"]}"/></linearGradient>'
        f'<linearGradient id="{gid}r" x1="0" x2="1"><stop offset="0" stop-color="{c["gold"]}"/>'
        f'<stop offset="1" stop-color="{c["gold"]}" stop-opacity="0"/></linearGradient></defs>'
        f'<rect x="{cx-half}" y="{cy-0.5}" width="{half-16}" height="1" fill="url(#{gid}l)"/>'
        f'<rect x="{cx+16}" y="{cy-0.5}" width="{half-16}" height="1" fill="url(#{gid}r)"/>'
        f'<path d="M{cx} {cy-5} L{cx+5} {cy} L{cx} {cy+5} L{cx-5} {cy} Z" fill="{c["accent"]}"/>'
        f'<circle cx="{cx-11}" cy="{cy}" r="1.6" fill="{c["gold"]}"/>'
        f'<circle cx="{cx+11}" cy="{cy}" r="1.6" fill="{c["gold"]}"/>'
    )

# ───────────────────────── 装饰图标（均以 (0,0) 为中心，约 ±16px）─────────────────────────
def _leaves(c, y=6):
    """左右两片线描叶子"""
    return (f'<path d="M-4 {y} Q-14 {y-2} -18 {y+6} Q-10 {y+8} -4 {y} Z" fill="none" stroke="{c["gold"]}" stroke-width="1.1"/>'
            f'<path d="M4 {y} Q14 {y-2} 18 {y+6} Q10 {y+8} 4 {y} Z" fill="none" stroke="{c["gold"]}" stroke-width="1.1"/>'
            f'<path d="M-6 {y+1.5} L-15 {y+5} M6 {y+1.5} L15 {y+5}" stroke="{c["gold"]}" stroke-width=".7"/>')

def icon_camellia(c):
    """山茶（椿）：五片圆瓣 + 金色花蕊 —— 插画中发饰即为山茶"""
    g = _leaves(c, 4)
    for k in range(5):
        a = math.radians(-90 + 72 * k)
        g += f'<circle cx="{6*math.cos(a):.2f}" cy="{6*math.sin(a)-2:.2f}" r="6.2" fill="{c["accent"]}" stroke="{c["deep"]}" stroke-width=".7"/>'
    g += f'<circle cx="0" cy="-2" r="3.6" fill="{c["gold"]}"/>'
    for k in range(6):
        a = math.radians(60 * k)
        g += f'<circle cx="{2*math.cos(a):.2f}" cy="{2*math.sin(a)-2:.2f}" r=".8" fill="{c["deep"]}"/>'
    return g

def icon_sakura(c):
    """樱：五片带缺口的花瓣"""
    petal = "M0 0 C-5 -4 -6 -10 -2.6 -13.5 L0 -11.4 L2.6 -13.5 C6 -10 5 -4 0 0 Z"
    g = "".join(f'<path d="{petal}" transform="translate(0 -1) rotate({72*k})" fill="{c["petal"]}" stroke="{c["accent"]}" stroke-width=".7"/>' for k in range(5))
    g += f'<circle cx="0" cy="-1" r="2" fill="{c["gold"]}"/>'
    return g

def icon_rose(c):
    """蔷薇：螺旋花心 + 叶"""
    return (_leaves(c, 5)
            + f'<circle cx="0" cy="-2" r="9" fill="{c["accent"]}" stroke="{c["deep"]}" stroke-width=".8"/>'
            + f'<path d="M0 -4 C3 -4 3 0 0 0 C-4 0 -4 -7 0 -7 C6 -7 6 2 0 3 C-7 4 -8 -9 0 -10" '
              f'fill="none" stroke="{c["deep"]}" stroke-width="1.1" stroke-linecap="round"/>')

def icon_osmanthus(c):
    """桂花：成簇的四瓣小花"""
    g = _leaves(c, 6)
    for (x, y) in [(-6, 0), (4, -3), (0, 5), (7, 3), (-2, -6)]:
        for k in range(4):
            a = math.radians(45 + 90 * k)
            g += f'<circle cx="{x+2.1*math.cos(a):.2f}" cy="{y+2.1*math.sin(a):.2f}" r="1.9" fill="{c["gold"]}"/>'
        g += f'<circle cx="{x}" cy="{y}" r=".9" fill="{c["accent"]}"/>'
    return g

def icon_birdcage(c):
    """鸟笼：圆顶 + 竖栏 + 底座，栏内一只小鸟剪影"""
    st = f'stroke="{c["gold"]}" stroke-width="1.3" fill="none" stroke-linecap="round"'
    bars = "".join(f'<path d="M{x} 12 L{x} 0 Q{x*0.55:.1f} -11 0 -12.5" {st}/>' for x in (-10, -5, 5, 10))
    return (f'<circle cx="0" cy="-16" r="2.4" {st}/>'
            f'<path d="M0 -13.6 L0 12" {st}/>' + bars +
            f'<path d="M-10 4 L10 4" {st}/>'
            f'<rect x="-12.5" y="12" width="25" height="3" rx="1" fill="{c["gold"]}"/>'
            f'<path class="bird" d="M-2.5 8.5 C-2.5 5.5 1 5 2.5 6.5 L5 5.5 L3.6 7.6 C3.6 10 0 10.6 -2.5 8.5 Z" fill="{c["accent"]}"/>')

def icon_candelabra(c):
    """三枝烛台：卷曲烛臂 + 蜡烛；白天熄灭（只有烛芯），夜晚点亮（火苗摇曳 + 光晕）"""
    st = f'stroke="{c["gold"]}" stroke-width="1.3" fill="none" stroke-linecap="round"'
    g = ""
    if c["lit"]:   # 烛光光晕，画在最底层
        g += ('<defs><radialGradient id="halo"><stop offset="0" stop-color="#f0b46a" stop-opacity=".55"/>'
              '<stop offset="1" stop-color="#f0b46a" stop-opacity="0"/></radialGradient></defs>'
              '<ellipse class="glow" cx="0" cy="-14" rx="30" ry="20" fill="url(#halo)"/>')
    g += (f'<path d="M0 14 L0 -3" {st}/>'
         f'<path d="M-7 15 Q0 11 7 15" {st}/>'
         f'<path d="M0 5 C-3 6 -11 7 -11 -1" {st}/><path d="M0 5 C3 6 11 7 11 -1" {st}/>'
         f'<circle cx="0" cy="8" r="1.6" fill="{c["gold"]}"/>')
    for x, top, h in [(-11, -11, 10), (0, -15, 12), (11, -11, 10)]:
        g += f'<rect x="{x-1.8}" y="{top}" width="3.6" height="{h}" rx=".6" fill="{c["candle"]}" stroke="{c["gold"]}" stroke-width=".8"/>'
        if c["lit"]:   # 三朵火苗错开节奏
            delay = {-11: "-1.1s", 0: "0s", 11: "-2.3s"}[x]
            g += (f'<path class="flame" style="animation-delay:{delay}" d="M{x} {top-.8} C{x-2.4} {top-2.2} {x-1.2} {top-4.2} {x} {top-5.6} '
                  f'C{x+1.2} {top-4.2} {x+2.4} {top-2.2} {x} {top-.8} Z" fill="#e6a35c"/>'
                  f'<path class="flame" style="animation-delay:{delay}" d="M{x} {top-1} C{x-1.1} {top-1.8} {x-.6} {top-3} {x} {top-3.8} '
                  f'C{x+.6} {top-3} {x+1.1} {top-1.8} {x} {top-1} Z" fill="#fbe3b5"/>')
        else:          # 熄灭的烛芯
            g += f'<path d="M{x} {top} L{x} {top-2.2}" stroke="{c["ink"]}" stroke-width=".8" stroke-linecap="round"/>'
    return g

def icon_parasol(c):
    """白天：蕾丝阳伞（哥特洛丽塔的日间随身物）—— 圆顶伞面 + 伞骨 + 波浪蕾丝边，
    伞柄弯钩上系一枚酒红小蝴蝶结；整把伞轻轻摇摆。夜晚换成点亮的烛台。"""
    st = f'stroke="{c["gold"]}" stroke-width="1.2" fill="none" stroke-linecap="round" stroke-linejoin="round"'
    edge = "".join(f" Q{x-3.75} 1.2 {x-7.5} -3" for x in (15, 7.5, 0, -7.5))   # 伞面右端绕回左端：四段波浪蕾丝边
    canopy = f"M-15 -3 C-14 -11 -7 -15 0 -15 C7 -15 14 -11 15 -3{edge}"
    g = (f'<path d="{canopy} Z" fill="{c["bg"]}"/>'
         f'<path d="M0 -3 L0 11.5 C0 15.5 -5.5 15.5 -5.5 12" {st}/>'                     # 伞柄 + 弯钩
         f'<path d="{canopy}" {st}/>'
         + "".join(f'<path d="M0 -15 Q{x*0.62:.2f} -9.5 {x} -3" stroke="{c["gold"]}" stroke-width=".75" fill="none"/>'
                   for x in (-7.5, 7.5))
         + f'<path d="M0 -15 L0 -17.5" {st}/><circle cx="0" cy="-18.4" r="1" fill="{c["gold"]}"/>'
         f'<path d="M0 3 C-4.2 0.4 -4.6 5.4 0 3 Z M0 3 C4.2 0.4 4.6 5.4 0 3 Z" fill="{c["accent"]}"/>'   # 蝴蝶结
         f'<path d="M0 3 L-2 7.4 M0 3 L2 7.2" stroke="{c["accent"]}" stroke-width=".9" stroke-linecap="round"/>'
         f'<circle cx="0" cy="3" r=".9" fill="{c["deep"]}"/>')
    return f'<g class="parasol">{g}</g>'

# 季节花：冬 山茶 / 春 樱 / 夏 蔷薇 / 秋 桂花（按台北时间月份）
SEASON = {12: "camellia", 1: "camellia", 2: "camellia", 3: "sakura", 4: "sakura", 5: "sakura",
          6: "rose", 7: "rose", 8: "rose", 9: "osmanthus", 10: "osmanthus", 11: "osmanthus"}
ICONS = dict(camellia=icon_camellia, sakura=icon_sakura, rose=icon_rose, osmanthus=icon_osmanthus,
             birdcage=icon_birdcage, candelabra=icon_candelabra, parasol=icon_parasol)

BIG = {"birdcage": 1.5, "candelabra": 1.5, "parasol": 1.5}   # 有动画的图标放大，动作才看得清

def motif_divider(cx, cy, c, motif, half=200):
    """两端渐隐金线 + 中央图标 + 图标两侧小圆点"""
    gid = f"d{motif}"
    k = BIG.get(motif, 1)
    gap = 30 * k
    return (
        f'<defs><linearGradient id="{gid}l" x1="0" x2="1"><stop offset="0" stop-color="{c["gold"]}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{c["gold"]}"/></linearGradient>'
        f'<linearGradient id="{gid}r" x1="0" x2="1"><stop offset="0" stop-color="{c["gold"]}"/>'
        f'<stop offset="1" stop-color="{c["gold"]}" stop-opacity="0"/></linearGradient></defs>'
        f'<rect x="{cx-half}" y="{cy-.5}" width="{half-gap}" height="1" fill="url(#{gid}l)"/>'
        f'<rect x="{cx+gap}" y="{cy-.5}" width="{half-gap}" height="1" fill="url(#{gid}r)"/>'
        f'<circle cx="{cx-gap+5}" cy="{cy}" r="1.6" fill="{c["gold"]}"/>'
        f'<circle cx="{cx+gap-5}" cy="{cy}" r="1.6" fill="{c["gold"]}"/>'
        f'<g transform="translate({cx} {cy}) scale({k})">{ICONS[motif](c)}</g>'
    )

# ───────────────────────── 结尾装饰：红茶杯 ─────────────────────────
def teacup(cx, cy, c, k=2.0):
    """古金线描茶杯与茶托（与鸟笼、烛台同一线稿语言），杯中是酒红色的红茶，
    杯身一枚红心花色呼应扑克牌底纹；三缕热气错开节奏升起 —— 谢客时奉上一杯茶。"""
    st = f'stroke="{c["gold"]}" stroke-width="1" fill="none" stroke-linecap="round" stroke-linejoin="round"'
    heart = "M0 2.6 C-3.4 0.2 -3.2 -2.4 -1.5 -2.4 C-0.7 -2.4 -0.2 -1.9 0 -1.3 C0.2 -1.9 0.7 -2.4 1.5 -2.4 C3.2 -2.4 3.4 0.2 0 2.6 Z"
    g = ''.join(
        f'<path class="steam" style="animation-delay:{dl}" d="{d}" stroke="{c["gold"]}" stroke-width=".8" '
        f'fill="none" stroke-linecap="round"/>'
        for d, dl in (("M-4.5 -7 C-7.5 -10.5 -1.5 -13.5 -4.5 -18", "0s"),
                      ("M0.5 -7.5 C-2.5 -11.5 3.5 -15 0.5 -21", "-1.2s"),
                      ("M5.5 -7 C2.5 -10.5 8.5 -13.5 5.5 -17.5", "-2.4s")))
    g += (f'<ellipse cx="0" cy="13" rx="18" ry="3.4" fill="{c["bg"]}"/>'          # 底色：挡住背后的花色底纹
          f'<path d="M-11.5 -3 C-11.5 5.5 -6.5 11.4 0 11.4 C6.5 11.4 11.5 5.5 11.5 -3 Z" fill="{c["bg"]}"/>'
          f'<ellipse cx="0" cy="13" rx="18" ry="3.4" {st}/>'                     # 茶托
          f'<ellipse cx="0" cy="12.6" rx="9.5" ry="1.7" stroke="{c["gold"]}" stroke-width=".6" fill="none"/>'
          f'<path d="M-11.5 -3 C-11.5 5.5 -6.5 11.4 0 11.4 C6.5 11.4 11.5 5.5 11.5 -3" {st}/>'   # 杯身
          f'<path d="M11.2 0 C17 -1.2 17.6 6 10.4 6.6" {st}/>'                      # 杯柄
          f'<path d="M11.4 2 C14.4 1.6 14.6 4.6 10.9 5" stroke="{c["gold"]}" stroke-width=".6" fill="none"/>'
          f'<path d="M-10.8 1.2 C-6 2.8 6 2.8 10.8 1.2" stroke="{c["gold"]}" stroke-width=".55" fill="none"/>'  # 金边
          f'<ellipse cx="0" cy="-3" rx="11.5" ry="2.5" {st}/>'                      # 杯口
          f'<ellipse cx="0" cy="-2.7" rx="9.6" ry="1.6" fill="{c["accent"]}" opacity=".9"/>'     # 红茶
          f'<path d="{heart}" transform="translate(0 5.4)" fill="{c["accent"]}"/>')
    return f'<g transform="translate({cx} {cy}) scale({k})">{g}</g>'

def est_width(s, size, spacing=0):
    """粗略估算文字宽度：CJK/假名按 1em，其余按 0.55em，外加字距"""
    w = sum(size if ord(ch) > 0x2E7F else size * 0.55 for ch in s)
    return w + spacing * len(s)

def flourish(x, y, c, d=1):
    """
    卷须：靠近文字一端是小涡卷，向外延伸成细线，末端一片小叶 + 圆点。
    d = 1 向右延伸，d = -1 向左（镜像）
    """
    g = c["gold"]
    return (f'<g transform="translate({x:.1f} {y}) scale({d} 1)" fill="none" stroke="{g}" stroke-linecap="round">'
            f'<path d="M6 0 C2 -6 -6 -4 -4 1 C-3 4 1 3 0 0" stroke-width="1.1"/>'
            f'<path d="M6 0 C20 -5 34 4 62 0" stroke-width="1"/>'
            f'<path d="M30 0 C34 -7 42 -8 46 -6 C42 -2 36 -1 30 0 Z" fill="{g}" stroke="none" opacity=".8"/>'
            f'<circle cx="66" cy="0" r="1.5" fill="{g}" stroke="none"/></g>')

def section_label(cx, y, s, c):
    """小节标题：字距拉开的金色小字，两侧卷须随文字宽度自适应"""
    half = est_width(s, 15, 4) / 2 + 12
    return (text(cx, y, s, 15, c["gold"], "m", spacing=4)
            + flourish(cx + half, y - 5, c, 1)
            + flourish(cx - half, y - 5, c, -1))

# ───────────────────────── 文字动画 ─────────────────────────
_adv_cache: dict = {}

def advance(ch, size, weight="Medium"):
    """从字体子集读取字形前进宽度，用于逐字母精确排版"""
    if weight not in _adv_cache:
        f = TTFont(FONT_DIR / f"LXGWWenKai-{weight}.subset.woff2")
        _adv_cache[weight] = (f.getBestCmap(), f["hmtx"].metrics, f["head"].unitsPerEm)
    cmap, hmtx, upm = _adv_cache[weight]
    g = cmap.get(ord(ch))
    return (hmtx[g][0] if g else upm * .5) * size / upm

def wave_text(cx, y, s, size, fill, cls="m", spacing=0, weight="Medium"):
    """
    逐字母排列的标题：每个字母是独立的 <g class="ltr">，依次延迟形成波浪跳动；
    每个字母再叠一层流光（fill=url(#sheen)），跳动时流光随字母一起移动
    """
    widths = [advance(ch, size, weight) for ch in s]
    total = sum(widths) + spacing * (len(s) - 1)
    x = cx - total / 2
    out = ""
    for i, (ch, w) in enumerate(zip(s, widths)):
        out += (f'<g class="ltr" style="--i:{i}">'
                f'<text x="{x:.1f}" y="{y}" class="{cls}" font-size="{size}" fill="{fill}">{esc(ch)}</text>'
                f'<text x="{x:.1f}" y="{y}" class="{cls}" font-size="{size}" fill="url(#sheen)">{esc(ch)}</text></g>')
        x += w + spacing
    return out

def sheen(x, y, s, size, cls="r", anchor="middle", spacing=0):
    """在已有文字上叠一层同样的文字，填充为移动的金色流光渐变"""
    return text(x, y, s, size, "url(#sheen)", cls, anchor, spacing)

def sheen_defs(H, c):
    """
    全卡共用的流光渐变：一条倾斜的亮带每 5 秒从左扫到右（SMIL 动画，<img> 中可播放）
    userSpaceOnUse 使各段文字共享同一道光，看起来是一束光扫过整张卡片
    """
    return ('<linearGradient id="sheen" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="140" y2="40">'
            f'<stop offset="0" stop-color="{c["sheen"]}" stop-opacity="0"/>'
            f'<stop offset=".5" stop-color="{c["sheen"]}" stop-opacity=".95"/>'
            f'<stop offset="1" stop-color="{c["sheen"]}" stop-opacity="0"/>'
            f'<animateTransform attributeName="gradientTransform" type="translate" '
            f'values="-260 0;{W+120} 0;{W+120} 0" keyTimes="0;.55;1" dur="5s" repeatCount="indefinite"/>'
            '</linearGradient>')

# ───────────────────────── 静态块（中 / 英 / 日 三语）─────────────────────────
# 修改文字后，若新增了汉字，需要重新生成 assets/fonts 下的字体子集（见 README 注释）
NAME = "REIJI007"
GREETING = "你好，欢迎 · Hello, welcome · こんにちは、ようこそ"
TAGLINE = (
    "互联网上一个安静的小角落。",
    "A quiet little corner of the internet.",
    "インターネットの、静かな片隅。",
)
ABOUT_LABEL = "简介 · About · 紹介"
ABOUT = (
    "一个喜欢绘画、作曲、编程的Z世代。",
    "A Gen Z who loves drawing, composing, and coding.",
    "絵を描くこと、曲を作ること、コードを書くことが好きなZ世代。",
)
MOTTO = "好きなものを、好きなかたちで。"
FOOTER = "感谢来访 · Thanks for stopping by · またね。"

def rose_window(cx, cy, c, R=96):
    """哥特玫瑰窗：同心圆 + 16 瓣尖拱花窗 + 内圈 8 瓣，极淡，作为标题背景"""
    g = c["gold"]
    out = f'<g opacity="{c["window"]}"><g class="window" fill="none" stroke="{g}">'
    out += f'<circle cx="{cx}" cy="{cy}" r="{R}" stroke-width="1.4"/><circle cx="{cx}" cy="{cy}" r="{R-7}" stroke-width=".7"/>'
    out += f'<circle cx="{cx}" cy="{cy}" r="{R*.42:.1f}" stroke-width="1"/><circle cx="{cx}" cy="{cy}" r="{R*.16:.1f}" stroke-width="1"/>'
    for k in range(16):   # 外圈尖拱花瓣
        a = 360 / 16 * k
        out += (f'<path d="M0 {-R*.44:.1f} C{R*.13:.1f} {-R*.58:.1f} {R*.11:.1f} {-R*.8:.1f} 0 {-(R-8):.1f} '
                f'C{-R*.11:.1f} {-R*.8:.1f} {-R*.13:.1f} {-R*.58:.1f} 0 {-R*.44:.1f} Z" '
                f'transform="translate({cx} {cy}) rotate({a})" stroke-width=".9"/>')
    for k in range(8):    # 内圈小瓣
        a = 360 / 8 * k + 22.5
        out += (f'<path d="M0 {-R*.17:.1f} C{R*.09:.1f} {-R*.24:.1f} {R*.08:.1f} {-R*.36:.1f} 0 {-R*.41:.1f} '
                f'C{-R*.08:.1f} {-R*.36:.1f} {-R*.09:.1f} {-R*.24:.1f} 0 {-R*.17:.1f} Z" '
                f'transform="translate({cx} {cy}) rotate({a})" stroke-width=".8"/>')
    return out + "</g></g>"

def render_header(c):
    body = (rose_window(W/2, 92, c)
            + wave_text(W/2, 78, NAME, 60, c["ink"], "m", spacing=6)
            + text(W/2, 122, GREETING, 19, c["accent"]) + sheen(W/2, 122, GREETING, 19))
    for i, line in enumerate(TAGLINE):
        body += text(W/2, 160 + i * 26, line, 15, c["sub"], spacing=0.5)
    return part(226, body, {"Medium": NAME, "Regular": GREETING + "".join(TAGLINE)},
               " / ".join((NAME, GREETING) + TAGLINE))

def render_divider(c, motif):
    h = 72 if motif in BIG else 48
    return part(h, motif_divider(W/2, h/2, c, motif), {}, f"divider: {motif}")

def render_about(c):
    zh, en, ja = ABOUT
    body = (section_label(W/2, 30, ABOUT_LABEL, c)
            + text(W/2, 82, zh, 26, c["ink"]) + sheen(W/2, 82, zh, 26)
            + text(W/2, 118, en, 18, c["ink"], opacity=.9)
            + text(W/2, 150, ja, 18, c["ink"], opacity=.9)
            + text(W/2, 190, MOTTO, 14, c["gold"], spacing=2))
    return part(210, body, {"Medium": ABOUT_LABEL, "Regular": "".join(ABOUT) + MOTTO},
               " / ".join(ABOUT + (MOTTO,)))

def render_footer(c):
    body = teacup(W/2, 46, c) + text(W/2, 106, FOOTER, 15, c["sub"], spacing=1)
    return part(122, body, {"Regular": FOOTER}, FOOTER)

# ───────────────────────── 动态 stats ─────────────────────────
QUERY = """query($login:String!){user(login:$login){
  followers{totalCount}
  repositories(ownerAffiliations:OWNER,isFork:false,privacy:PUBLIC,first:100){
    totalCount nodes{stargazerCount}}
  contributionsCollection{
    totalCommitContributions restrictedContributionsCount
    totalPullRequestContributions totalIssueContributions totalPullRequestReviewContributions
    contributionCalendar{totalContributions weeks{contributionDays{contributionCount date weekday}}}
  }
}}"""

def fetch_stats(token):
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": QUERY, "variables": {"login": USER}}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    if "errors" in d:
        raise RuntimeError(d["errors"])
    u = d["data"]["user"]
    cc = u["contributionsCollection"]
    cal = cc["contributionCalendar"]
    days = [dd for week in cal["weeks"] for dd in week["contributionDays"]]
    active_dates = [datetime.date.fromisoformat(dd["date"]) for dd in days if dd["contributionCount"] > 0]
    max_streak = streak = 0
    previous = None
    for day in active_dates:
        streak = streak + 1 if previous and (day - previous).days == 1 else 1
        max_streak = max(max_streak, streak)
        previous = day
    current_streak = 0
    if active_dates and (TODAY - active_dates[-1]).days <= 1:
        current_streak = 1
        for i in range(len(active_dates) - 1, 0, -1):
            if (active_dates[i] - active_dates[i - 1]).days != 1:
                break
            current_streak += 1
    repos = u["repositories"]
    return dict(
        stars=sum(n["stargazerCount"] for n in repos["nodes"]),
        commits=cc["totalCommitContributions"] + cc["restrictedContributionsCount"],
        prs=cc["totalPullRequestContributions"],
        issues=cc["totalIssueContributions"],
        reviews=cc["totalPullRequestReviewContributions"],
        repos=repos["totalCount"],
        contrib=cal["totalContributions"],
        followers=u["followers"]["totalCount"],
        active_days=len(active_dates),
        current_streak=current_streak,
        max_streak=max_streak,
        rank_ready=True,
        weeks=[[(dd["contributionCount"], dd["weekday"]) for dd in w["contributionDays"]] for w in cal["weeks"]],
    )

def mock_stats():
    import random
    random.seed(7)
    weeks = [[(random.choice([0, 0, 0, 1, 2, 3, 5, 8]), d) for d in range(7)] for _ in range(53)]
    return dict(stars=128, commits=342, prs=31, issues=18, reviews=7, repos=16,
                contrib=sum(n for w in weeks for n, _ in w), followers=24,
                active_days=sum(1 for w in weeks for n, _ in w if n > 0),
                current_streak=5, max_streak=12,
                rank_ready=True, weeks=weeks)

def fmt(n):
    return f"{n/1000:.1f}k".replace(".0k", "k") if n >= 1000 else str(n)

def levels(weeks):
    """按非零贡献的四分位切成 5 档（与 GitHub 自身的分档逻辑一致）"""
    vals = sorted(n for w in weeks for n, _ in w if n > 0)
    if not vals:
        return lambda n: 0
    q = [vals[int(len(vals) * p) - 1 if int(len(vals) * p) > 0 else 0] for p in (.25, .5, .75)]
    return lambda n: 0 if n == 0 else 1 + sum(n > t for t in q)

def exponential_cdf(x):
    """GitHub Readme Stats rank reference: exponential cumulative distribution."""
    return 1 - 2 ** (-x)

def log_normal_cdf(x):
    """Lightweight log-normal CDF approximation used by GitHub Readme Stats."""
    return x / (1 + x)

def calculate_rank(s):
    """
    Reference implementation of the GitHub Readme Stats weighted rank model.
    This profile uses the latest 12-month contribution window, not an official
    GitHub-issued score. Lower top_percentile means a stronger estimated tier.
    """
    weights = {"commits": 2, "prs": 3, "issues": 1, "reviews": 1, "stars": 4, "followers": 1}
    medians = {"commits": 250, "prs": 50, "issues": 25, "reviews": 2, "stars": 50, "followers": 10}
    total_weight = sum(weights.values())
    score = (
        weights["commits"] * exponential_cdf(s["commits"] / medians["commits"])
        + weights["prs"] * exponential_cdf(s["prs"] / medians["prs"])
        + weights["issues"] * exponential_cdf(s["issues"] / medians["issues"])
        + weights["reviews"] * exponential_cdf(s["reviews"] / medians["reviews"])
        + weights["stars"] * log_normal_cdf(s["stars"] / medians["stars"])
        + weights["followers"] * log_normal_cdf(s["followers"] / medians["followers"])
    )
    top_percentile = (1 - score / total_weight) * 100
    thresholds = [1, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    levels = ["S", "A+", "A", "A-", "B+", "B", "B-", "C+", "C"]
    index = next((i for i, threshold in enumerate(thresholds) if top_percentile <= threshold), 8)
    return levels[index], max(0, min(100, top_percentile))

def render_stats(c, s, today):
    # 评级所需字段齐全时才显示等级，避免旧缓存被误判为低等级。
    rank_ready = s.get("rank_ready") and all(k in s for k in ("prs", "issues", "reviews", "stars", "followers", "commits"))
    if rank_ready:
        rank, top_percentile = calculate_rank(s)
        rank_label = f"EST. TOP {top_percentile:.1f}%"
        progress = max(0, min(100, 100 - top_percentile))
    else:
        rank, rank_label, progress = "—", "DATA PENDING", 0

    body = section_label(W/2, 30, "GitHub Statistics", c)
    # 左侧等级徽章：圆环代表反向百分位，数值越小评级越高。
    cx, cy, radius = 112, 143, 49
    circumference = 2 * math.pi * radius
    dash = circumference * progress / 100
    body += (f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{c["gold"]}" '
             f'stroke-width="3" opacity=".2"/>'
             f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{c["gold"]}" '
             f'stroke-width="3.5" stroke-linecap="round" stroke-dasharray="{dash:.1f} {circumference:.1f}" '
             f'transform="rotate(-90 {cx} {cy})"/>')
    body += text(cx, 78, "PROFILE RANK", 12, c["sub"], spacing=2)
    body += text(cx, 158, rank, 48, c["ink"], "m")
    body += text(cx, 218, rank_label, 13, c["gold"], "m", spacing=1)
    body += text(cx, 238, "MODEL ESTIMATE · NOT OFFICIAL", 8, c["sub"], spacing=.4, opacity=.8)

    # 右侧三列 × 三行，避免单行塞入过多数据。
    items = [
        (fmt(s.get("stars", 0)), "Stars", "星标"),
        (fmt(s.get("commits", 0)), "Commits", "提交"),
        (fmt(s.get("prs", 0)), "Pull Requests", "合并请求"),
        (fmt(s.get("issues", 0)), "Issues", "议题"),
        (fmt(s.get("reviews", 0)), "Reviews", "代码审查"),
        (fmt(s.get("followers", 0)), "Followers", "关注者"),
        (fmt(s.get("repos", 0)), "Public Repos", "公开仓库"),
        (fmt(s.get("active_days", 0)), "Active Days", "活跃天数"),
        (f'{s.get("current_streak", 0)}d', "Current Streak", "连续贡献"),
    ]
    xs, row_ys = [315, 500, 685], [91, 160, 229]
    nums = labels = ""
    for i, (value, label, sub) in enumerate(items):
        row, col = divmod(i, 3)
        x, y = xs[col], row_ys[row]
        body += f'<g class="hop">{text(x, y, value, 29, c["ink"], "m")}{sheen(x, y, value, 29, "m")}</g>'
        body += text(x, y + 23, label, 12, c["sub"], spacing=.5)
        body += text(x, y + 39, sub, 10, c["sub"], spacing=.5, opacity=.78)
        nums += value
        labels += label + sub

    # 贡献热力图：53 周 × 7 天，圆角小方块。
    cell, gap = 11, 3.6
    gw = len(s["weeks"]) * (cell + gap) - gap
    x0, y0 = (W - gw) / 2, 278
    lv = levels(s["weeks"])
    import random
    rnd = random.Random(7)
    SPARK_O = [0, .35, .5, .68, .9]
    HALO_O = [0, .25, .4, .58, .8]
    halos = cells = sparks = ""
    hot = []
    for wi, week in enumerate(s["weeks"]):
        for n, wd in week:
            x, y = x0 + wi*(cell+gap), y0 + wd*(cell+gap)
            l = lv(n)
            cells += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" rx="2.5" '
                      f'fill="{c["heat"][l]}"/>')
            if l == 0:
                continue
            dur = rnd.uniform(1.8, 3.6)
            st = f'--d:{dur:.1f}s;--l:{-rnd.uniform(0, dur):.1f}s'
            cx, cy = x + cell/2, y + cell/2
            halos += (f'<circle class="halo" cx="{cx:.1f}" cy="{cy:.1f}" r="{cell*1.1:.1f}" fill="url(#heatglow)" '
                      f'style="{st};--o:{HALO_O[l]}"/>')
            sparks += (f'<rect class="spark" x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" rx="2.5" '
                       f'fill="#fff6e0" style="{st};--o:{SPARK_O[l]}"/>')
            hot.append((cx, cy, l))
    top = max((l for _, _, l in hot), default=0)
    pool = [h for h in hot if h[2] >= max(top - 1, 1)] if top else []
    stars = ""
    for cx, cy, _ in rnd.sample(pool, min(14, len(pool))):
        dur = rnd.uniform(4, 7)
        a, w = 10, 1.6
        stars += (f'<path class="glint" d="M{cx:.1f} {cy-a:.1f} Q{cx+w:.1f} {cy-w:.1f} {cx+a:.1f} {cy:.1f} '
                  f'Q{cx+w:.1f} {cy+w:.1f} {cx:.1f} {cy+a:.1f} Q{cx-w:.1f} {cy+w:.1f} {cx-a:.1f} {cy:.1f} '
                  f'Q{cx-w:.1f} {cy-w:.1f} {cx:.1f} {cy-a:.1f} Z" fill="#fffaf0" stroke="#d9a441" stroke-width=".7" '
                  f'style="--d:{dur:.1f}s;--l:{-rnd.uniform(0, dur):.1f}s"/>')
    body += ('<defs><radialGradient id="heatglow"><stop offset="0" stop-color="#f2b866"/>'
             '<stop offset="1" stop-color="#f2b866" stop-opacity="0"/></radialGradient></defs>'
             + halos + cells + sparks + stars)

    title = f"GitHub rank {rank}, {rank_label}; " + ", ".join(f"{lab} {v}" for v, lab, _ in items)
    # 热力图下方不再放说明行；压缩模块高度，让卡片更紧凑。
    return part(int(y0 + 7*(cell+gap) + 18), body,
                {"Medium": "GitHub" + rank + nums, "Regular": labels + rank_label}, title)

def part(height, body, texts, title):
    """一个可拼接的片段：高度 + SVG 内容 + 用到的文字（按字重）+ 无障碍标题"""
    return dict(h=height, body=body, texts=texts, title=title)

def playing_card_pattern(c):
    """四种扑克牌花色组成的连续底纹：黑桃、红心、梅花、方块。"""
    g = c["gold"]
    spade = ('<path d="M0 -18 C-5 -11 -15 -4 -15 3 C-15 10 -7 13 0 6 '
             'C7 13 15 10 15 3 C15 -4 5 -11 0 -18 Z"/>'
             '<path d="M-3 5 L-6 17 H6 L3 5 Z"/>')
    heart = '<path d="M0 16 C-4 12 -17 3 -17 -7 C-17 -17 -4 -20 0 -9 C4 -20 17 -17 17 -7 C17 3 4 12 0 16 Z"/>'
    club = ('<path d="M0 -17 C-5 -17 -8 -13 -8 -8 C-8 -5 -6 -3 -4 -2 '
            'C-11 -6 -17 -1 -17 5 C-17 12 -8 15 -2 8 L-4 18 H4 L2 8 '
            'C8 15 17 12 17 5 C17 -1 11 -6 4 -2 C6 -3 8 -5 8 -8 '
            'C8 -13 5 -17 0 -17 Z"/>')
    diamond = '<path d="M0 -19 L15 0 L0 19 L-15 0 Z"/>'
    icons = [
        (35, 35, spade), (105, 35, heart),
        (35, 105, club), (105, 105, diamond),
    ]
    marks = "".join(
        f'<g transform="translate({x} {y}) scale(.78)">{shape}</g>'
        for x, y, shape in icons
    )
    return (f'<pattern id="card-suits" width="140" height="140" patternUnits="userSpaceOnUse">'
            f'<g fill="{g}" stroke="{g}" stroke-width=".55" stroke-linejoin="round">{marks}</g>'
            f'</pattern>')


def corner_scroll(c):
    """卡片角饰：与横幅画框同款的卷草纹（左上角坐标，其余三角镜像复用）"""
    g = c["gold"]
    st = f'fill="none" stroke="{g}" stroke-linecap="round"'
    return (f'<g id="cardcorner">'
            f'<path d="M22 78 A56 56 0 0 1 78 22" {st} stroke-width="2.2"/>'
            f'<path d="M30 70 A40 40 0 0 1 70 30" {st} stroke-width="1"/>'
            f'<path d="M78 22 C102 20 118 30 112 44 C107 54 92 50 95 41 C97 35 105 36 104 41" {st} stroke-width="2"/>'
            f'<path d="M112 30 C128 22 146 24 160 22" {st} stroke-width="1.2"/>'
            f'<path d="M22 78 C20 102 30 118 44 112 C54 107 50 92 41 95 C35 97 36 105 41 104" {st} stroke-width="2"/>'
            f'<path d="M30 112 C22 128 24 146 22 160" {st} stroke-width="1.2"/>'
            f'<path d="M40 40 C54 44 66 56 70 70 C56 66 44 54 40 40 Z" fill="{g}"/>'
            f'<path d="M58 40 C66 42 72 46 74 52 C68 50 62 46 58 40 Z M40 58 C42 66 46 72 52 74 C50 68 46 62 40 58 Z" fill="{g}"/>'
            f'<path d="M22 13 L31 22 L22 31 L13 22 Z" fill="{c["accent"]}" stroke="{g}" stroke-width="1.4"/>'
            f'<circle cx="86" cy="58" r="2" fill="{g}"/><circle cx="58" cy="86" r="2" fill="{g}"/></g>')

def lace(y, x0, x1, c, down=True):
    """扇贝蕾丝：一排半圆弧 + 弧心小孔 + 弧间小点（洛丽塔裙摆褶边）"""
    g, r = c["gold"], 5
    n = int((x1 - x0) // (2 * r))
    x0 = (W - n * 2 * r) / 2
    sgn = 1 if down else -1
    d = f"M{x0} {y}"
    holes = ""
    for i in range(n):
        xa = x0 + i * 2 * r
        d += f" A{r} {r} 0 0 {0 if down else 1} {xa + 2*r} {y}"
        holes += f'<circle cx="{xa + r}" cy="{y + sgn*2.4}" r="1.1"/>'
        holes += f'<circle cx="{xa}" cy="{y + sgn*7.5}" r=".8"/>'
    return (f'<g opacity=".55"><path d="M{x0} {y} H{x0 + n*2*r}" stroke="{g}" stroke-width=".7"/>'
            f'<path d="{d}" fill="none" stroke="{g}" stroke-width=".9"/>'
            f'<g fill="{g}">{holes}</g></g>')

def compose_card(c, parts, pad=48):
    """
    把各片段竖向拼成一张自带底色的卡片。
    自带底色是「按时间切换」的前提：卡片不再依赖访客的 GitHub 主题。
    """
    y, body, texts = pad, "", {}
    for p in parts:
        body += f'<g transform="translate(0 {y})">{p["body"]}</g>'
        y += p["h"]
        for w, t in p["texts"].items():
            texts[w] = texts.get(w, "") + t
    H = int(y + pad)
    g = c["gold"]
    S = .6   # 角饰缩放：角饰内框线 22/30 × 0.6 ≈ 卡片双线 13/18
    frame = (f'<defs>{playing_card_pattern(c)}{corner_scroll(c)}{sheen_defs(H, c)}'
             f'<clipPath id="card-clip"><rect width="{W}" height="{H}" rx="14"/></clipPath>'
             # 流光遮罩必须和底纹用同一块图案、放在同一个漂移组里：之前遮罩是静止的，
             # 底纹却在缓慢平移+缩放，流光点亮的是"原位"的花色，与漂走的花色错开形成重影
             f'<mask id="suits-mask" maskUnits="userSpaceOnUse" mask-type="alpha">'
             f'<rect x="-100" y="-100" width="{W+200}" height="{H+200}" fill="url(#card-suits)"/></mask></defs>'
             f'<rect width="{W}" height="{H}" rx="14" fill="{c["bg"]}"/>'
             f'<g clip-path="url(#card-clip)">'
             f'<g class="suits-drift"><rect x="-100" y="-100" width="{W+200}" height="{H+200}" fill="url(#card-suits)" opacity="{c["suits"]}"/>'
             f'<g mask="url(#suits-mask)">'
             f'<circle class="suit-wave wave-a" cx="{W/2}" cy="{H/2}" r="310" stroke="{c["suit_glow"]}" stroke-opacity=".95"/>'
             f'<circle class="suit-wave wave-b" cx="{W/2}" cy="{H/2}" r="310" stroke="{c["suit_glow"]}" stroke-opacity=".85"/>'
             f'</g></g></g>'
             # 双线边框在四角断开，与角饰的圆弧端点（外 46.8 / 内 42，按 0.6 缩放后）衔接
             f'<path d="M47 13.2 H{W-47} M{W-13.2} 47 V{H-47} M{W-47} {H-13.2} H47 M13.2 {H-47} V47" '
             f'fill="none" stroke="{g}" stroke-width="1.3" opacity=".75"/>'
             f'<path d="M42 18 H{W-42} M{W-18} 42 V{H-42} M{W-42} {H-18} H42 M18 {H-42} V42" '
             f'fill="none" stroke="{g}" stroke-width=".6" opacity=".55"/>'
             + lace(18, 110, W - 110, c, down=True)
             + lace(H - 18, 110, W - 110, c, down=False)
             + motes(c, H)
             + f'<use href="#cardcorner" transform="scale({S})"/>'
             f'<use href="#cardcorner" transform="matrix({-S} 0 0 {S} {W} 0)"/>'
             f'<use href="#cardcorner" transform="matrix({S} 0 0 {-S} 0 {H})"/>'
             f'<use href="#cardcorner" transform="matrix({-S} 0 0 {-S} {W} {H})"/>')
    title = " / ".join(p["title"] for p in parts if not p["title"].startswith("divider"))
    return svg(H, frame + body + petals(c, H), texts, title)

# ───────────────────────── 主流程 ─────────────────────────
def write(name, content):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content, encoding="utf-8")
    print(f"  {name:<22} {len(content.encode())/1024:6.1f} KB")

STATS_CACHE = OUT / "stats.json"

def load_stats():
    """取最新 stats；失败时退回上次缓存，保证卡片总能完整渲染"""
    if "--mock" in sys.argv:
        return mock_stats()
    token = os.environ.get("GITHUB_TOKEN")
    try:
        if not token:
            raise RuntimeError("GITHUB_TOKEN 未设置")
        stats = fetch_stats(token)
        OUT.mkdir(parents=True, exist_ok=True)
        STATS_CACHE.write_text(json.dumps(stats), encoding="utf-8")
        return stats
    except Exception as e:
        if STATS_CACHE.exists():
            print(f"stats 获取失败，使用缓存：{e}")
            return json.loads(STATS_CACHE.read_text(encoding="utf-8"))
        print(f"stats 获取失败且无缓存，使用占位数据：{e}")
        return dict(stars=0, commits=0, contrib=0, followers=0,
                    weeks=[[(0, d) for d in range(7)] for _ in range(53)])

def main():
    stats = load_stats()
    today = TODAY.isoformat()
    cards = {}
    for mode, c in THEMES.items():
        cards[mode] = compose_card(c, [
            render_header(c),
            render_divider(c, SEASON[TODAY.month]),
            render_about(c),
            render_divider(c, "birdcage"),
            render_stats(c, stats, today),
            render_divider(c, "candelabra" if c["lit"] else "parasol"),   # 夜：点亮的烛台 / 昼：蕾丝阳伞
            render_footer(c),
        ])
        write(f"card-{mode}.svg", cards[mode])

if __name__ == "__main__":
    main()
