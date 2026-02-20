#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "╔══════════════════════════════════════════════════╗"
echo "║     🔧 HMMWV TECHNICAL ASSISTANT                ║"
echo "╚══════════════════════════════════════════════════╝"

# Create venv if needed
[ ! -d "venv" ] && python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

# Ensure directories
mkdir -p knowledge_base extracted_images chroma_db

PDF_COUNT=$(find knowledge_base -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
echo "📚 $PDF_COUNT PDF(s) in knowledge_base/"
echo "🚀 Launching → http://localhost:8501"

streamlit run app.py --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
