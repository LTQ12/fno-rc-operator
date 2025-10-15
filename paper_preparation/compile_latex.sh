#!/bin/bash

# LaTeX compilation script for FNO-RC paper (supports filename argument)

# Usage: bash compile_latex.sh fno_rc_paper_nmi.tex

MAIN_TEX=${1:-fno_rc_paper_nmi.tex}
BASE_NAME=${MAIN_TEX%.tex}

echo "🚀 开始编译LaTeX论文..."

# 检查必要文件
if [ ! -f "$MAIN_TEX" ]; then
    echo "❌ 错误: $MAIN_TEX 文件不存在"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo "❌ 错误: references.bib 文件不存在"
    exit 1
fi

# 清理之前的编译文件
echo "🧹 清理之前的编译文件..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# 第一次编译
echo "📝 第一次 pdflatex 编译... ($MAIN_TEX)"
pdflatex -interaction=nonstopmode "$MAIN_TEX"

# 编译参考文献
echo "📚 编译参考文献... ($BASE_NAME)"
bibtex "$BASE_NAME"

# 第二次编译 (解决引用)
echo "📝 第二次 pdflatex 编译... ($MAIN_TEX)"
pdflatex -interaction=nonstopmode "$MAIN_TEX"

# 第三次编译 (最终版本)
echo "📝 第三次 pdflatex 编译... ($MAIN_TEX)"
pdflatex -interaction=nonstopmode "$MAIN_TEX"

# 检查编译结果
if [ -f "$BASE_NAME.pdf" ]; then
    echo "✅ 编译成功! PDF文件已生成: $BASE_NAME.pdf"
    
    # 显示文件信息
    echo "📊 文件信息:"
    ls -lh "$BASE_NAME.pdf"
    
    # 显示页数
    if command -v pdfinfo >/dev/null 2>&1; then
      echo "📄 页数统计:"
      pdfinfo "$BASE_NAME.pdf" | grep Pages
    fi
    
    echo ""
    echo "🎉 LaTeX论文编译完成!"
    echo "📁 输出文件: $BASE_NAME.pdf"
    
else
    echo "❌ 编译失败! 请检查LaTeX日志文件"
    echo "💡 查看日志: cat $BASE_NAME.log"
    exit 1
fi

# 可选: 清理中间文件
read -p "🗑️  是否清理中间文件? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 清理中间文件..."
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz
    echo "✅ 清理完成!"
fi

echo "📖 现在可以打开 fno_rc_paper.pdf 查看论文!"
