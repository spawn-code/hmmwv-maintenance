"""Components package — individual modules are loaded by app.py via importlib."""

# When loaded by app.py's _load_module, the sub-modules are already in
# sys.modules, so these re-exports just work.
import sys

if "components.pdf_processor" in sys.modules:
    from components.pdf_processor import PDFProcessor
if "components.vector_store" in sys.modules:
    from components.vector_store import VectorStore
if "components.ai_engine" in sys.modules:
    from components.ai_engine import AIEngine

__all__ = ["PDFProcessor", "VectorStore", "AIEngine"]
