#!/usr/bin/env bash
# 重新生成霞鹜文楷基础子集（修改 render.py 文字、新增汉字后运行一次）
# 用法: bash scripts/build_fonts.sh /path/to/LXGWWenKai-Regular.ttf /path/to/LXGWWenKai-Medium.ttf
# 字体下载: https://github.com/lxgw/LxgwWenKai/releases
# 子集 = ASCII + 常用标点 + 假名 + render.py 中出现的全部字符
set -euo pipefail
cd "$(dirname "$0")/.."
for src in "$@"; do
  w=$(basename "$src" .ttf | sed 's/LXGWWenKai-//')
  pyftsubset "$src" \
    --unicodes="U+0020-007E,U+00B7,U+2014,U+2026,U+3000-303F,U+3040-309F,U+30A0-30FF,U+FF01-FF5E" \
    --text-file=scripts/render.py \
    --layout-features='kern,palt' --name-IDs='*' --flavor=woff2 \
    --output-file="assets/fonts/LXGWWenKai-$w.subset.woff2"
  echo "built assets/fonts/LXGWWenKai-$w.subset.woff2"
done
