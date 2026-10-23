"""vlmstudy: controlled diagnosis of the evidence VLMs use to classify Vietnamese news (BN_VN3S_MM).

Modules
-------
core    data loading, prompt, constrained scoring, input conditions, gradient x input relevance,
        deletion check, bootstrap statistics and the analysis tables/figures (RQ1-RQ3)
ocr     text printed inside the photograph: EasyOCR, text masking vs random masking, OCR-only input
figure  qualitative "where does the model look" figure (needs the local photographs)
"""
from . import core  # noqa: F401

__all__ = ["core"]
__version__ = "1.0.0"
