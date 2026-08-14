#!/bin/zsh
# 构建毕业论文，并把每一条不能被破坏的不变量打印出来。
#
# 为什么这个脚本在仓库里而不在 /tmp：它的前身放在会话临时目录里，被清理过一次；
# 更要紧的是，它前身的 latexmk 调用曾以 "command not found" 静默失败，脚本却继续
# 兴高采烈地报告了一份五小时前的 main.pdf —— 我把那份报告当成了通过的构建。
# 破绽是各章余量在明明会移动版面的编辑后逐字节不变。所以这里 PATH 自己设，
# 并把 latexmk 的存在设为硬闸。
#
# 用法:  scripts/build_dissertation.sh [--clean] [--no-figures]
# 退出码非 0 == 有不变量被破坏。可直接接进 CI 或 Makefile。
set -u
export PATH="/Library/TeX/texbin:$PATH"
command -v latexmk >/dev/null || { echo "FAIL: latexmk 不在 PATH 上"; exit 1; }

ROOT=${0:A:h:h}
D="$ROOT/writing/dissertation"
FAIL=0
note() { printf "  %-34s %s\n" "$1" "$2" }
bad()  { printf "  %-34s %s   <-- FAIL\n" "$1" "$2"; FAIL=1 }

# ---- 图的几何闸。放在构建之前，因为它拦的失败是静默的：用错 driver 重生成的图
#      照样渲染、照样能构建，LaTeX 一个警告都不报。
if [[ "${*}" != *--no-figures* ]]; then
  echo "=== 图：字号下限闸 ==="
  if python3 "$ROOT/scripts/analysis/diss_appendix_figs/audit_inclusion_geometry.py" --gate \
       > /tmp/diss_figgate.out 2>&1; then
    note "附录 E 印刷字号 >= 9pt" "PASS"
  else
    bad "附录 E 印刷字号 >= 9pt" "$(grep -m1 GATE /tmp/diss_figgate.out)"
  fi
fi

cd "$D" || exit 1
[[ "${*}" == *--clean* ]] && latexmk -C >/dev/null 2>&1

echo "=== 构建 ==="
if latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /tmp/diss_build.out 2>&1; then
  note "latexmk" "rc=0"
else
  bad "latexmk" "rc!=0"; grep -n "^!" /tmp/diss_build.out | head -10
fi

echo "=== 页面与上限 ==="
note "总页数" "$(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
# 印刷页码 != PDF 页序：前言用罗马数字，正文 p1 落在 PDF 第 1+offset 页。main.toc
# 记的是印刷页码，所以下面的章起始检查是对的；但手工 pdftotext -f/-l 用的是 PDF
# 页序。我曾按 PDF 页序量各章余量，整组数字全废。所以把偏移印出来。
python3 - <<'PY'
import re, subprocess
n = int(subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True)
        .stdout.split("Pages:")[1].split()[0])
for p in range(1, n + 1):
    h = subprocess.run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), "main.pdf", "-"],
                       capture_output=True, text=True).stdout.split("\n")[0]
    if "CHAPTER" in h or "APPENDIX" in h:
        nums = re.findall(r"\b(\d{1,3})\b", h)
        if nums:
            off = p - int(nums[-1])
            print(f"  {'印刷 p1 = PDF 页':<34}{1 + off}  (偏移 +{off}；"
                  f"正文 1..60 = PDF {1 + off}..{60 + off})")
            break
PY
python3 - "$D" <<'PY'
import re, sys
toc = open(sys.argv[1] + "/main.toc").read()
st = {m.group(1): int(m.group(2)) for m in
      re.finditer(r'contentsline \{chapter\}\{\\numberline \{([0-9])\}[^}]*\}\{(\d+)\}', toc)}
ref = re.search(r'contentsline \{chapter\}\{References\}\{(\d+)\}', toc)
base = {'1': 1, '2': 6, '3': 16, '4': 28, '5': 44, '6': 58}
ok = st == base
print(f"  {'章起始 1/6/16/28/44/58':<34}{'一致' if ok else '变动 ' + str(st)}"
      + ("" if ok else "   <-- FAIL"))
if ref:
    body = int(ref.group(1)) - 1
    print(f"  {'正文页数':<34}{body}/60"
          + ("" if body <= 60 else "   <-- FAIL 超出硬上限"))
sys.exit(0 if ok and ref and int(ref.group(1)) - 1 <= 60 else 3)
PY
[[ $? -ne 0 ]] && FAIL=1

echo "=== 引用与浮动体 ==="
for spec in "未定义引用:Reference \`" "未解析引文:Citation \`" \
            "重复标签:multiply defined" "重复 destination:destination with the same" \
            "overfull hbox:^Overfull \\\\hbox" "overfull vbox:^Overfull \\\\vbox" \
            "float 过大:Float too large" "geometry 警告:Package geometry Warning"; do
  lbl=${spec%%:*}; pat=${spec#*:}
  n=$(grep -c "$pat" main.log)
  [[ "$n" -eq 0 ]] && note "$lbl" "0" || bad "$lbl" "$n"
done
note "参考文献条数" "$(grep -c '^\\bibitem' main.bbl)"

echo
[[ $FAIL -eq 0 ]] && echo "全部不变量通过。" || echo "有不变量被破坏，见上面的 FAIL。"
exit $FAIL
