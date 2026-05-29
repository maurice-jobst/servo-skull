"""Phase 1: Deterministic document extraction."""
import logging
from pathlib import Path
from typing import Any

from servo_skull._utils import setup_logging
from servo_skull.models import DocumentExtract

logger = setup_logging(__name__)


class Extractor:
    """Handles deterministic extraction from various file types."""

    SUPPORTED_TYPES = {
        ".pdf": "pdf",
        ".docx": "office",
        ".xlsx": "office",
        ".pptx": "office",
        ".doc": "office",
        ".xls": "office",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".tiff": "image",
        ".gif": "image",
        ".mp3": "audio",
        ".wav": "audio",
        ".m4a": "audio",
        ".flac": "audio",
        ".txt": "text",
        ".md": "text",
        ".rst": "text",
        ".html": "office",
        ".htm": "office",
    }

    def extract(self, file_path: Path) -> DocumentExtract:
        """Extract text and structure from a file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {suffix}")

        doc_type = self.SUPPORTED_TYPES[suffix]
        logger.info(f"Extracting {doc_type} from {file_path.name}")

        if doc_type == "text":
            return self._extract_text(file_path)
        elif doc_type == "pdf":
            return self._extract_pdf(file_path)
        elif doc_type == "office":
            return self._extract_office(file_path)
        elif doc_type == "image":
            return self._extract_image(file_path)
        elif doc_type == "audio":
            return self._extract_audio(file_path)
        else:
            raise RuntimeError(f"Extraction not implemented for {doc_type}")

    def _extract_text(self, file_path: Path) -> DocumentExtract:
        """Extract from plain text files."""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        confidence = 0.99
        warnings: list[str] = []

        if len(text) == 0:
            warnings.append("File is empty")
            confidence = 0.5

        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0

        return DocumentExtract(
            original_filename=file_path.name,
            document_type="text",
            extracted_text=text,
            metadata={"encoding": "utf-8", "file_size_bytes": file_size},
            confidence=confidence,
            warnings=warnings,
            extraction_tool="text_reader",
        )

    def _extract_pdf(self, file_path: Path) -> DocumentExtract:
        """Extract from PDF using pymupdf4llm."""
        try:
            import pymupdf4llm
            text = pymupdf4llm.to_markdown(str(file_path))
            confidence = 0.95
            warnings: list[str] = []
        except ImportError:
            text = ""
            confidence = 0.0
            warnings: list[str] = ["pymupdf4llm not installed"]
        except Exception as e:
            logger.warning(f"PDF extraction error: {e}")
            text = ""
            confidence = 0.0
            warnings: list[str] = [str(e)]

        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0

        return DocumentExtract(
            original_filename=file_path.name,
            document_type="pdf",
            extracted_text=text,
            metadata={"file_size_bytes": file_size},
            confidence=confidence,
            warnings=warnings,
            extraction_tool="pymupdf4llm",
        )

    def _extract_office(self, file_path: Path) -> DocumentExtract:
        """Extract from Office documents (DOCX, XLSX, PPTX, HTML) using MarkItDown."""
        text = ""
        metadata: dict[str, Any] = {}
        warnings: list[str] = []
        confidence = 0.90
        extraction_tool = "markitdown"

        suffix = file_path.suffix.lower()

        try:
            from markitdown import MarkItDown
            md_converter = MarkItDown()
            result = md_converter.convert(str(file_path))
            text = result.text_content
        except Exception as e:
            logger.warning(f"MarkItDown extraction failed for {file_path.name}: {e}. Trying fallback.")
            warnings.append(f"MarkItDown failed: {e}")
            extraction_tool = f"office_parser_fallback({suffix})"
            
            # Fallback to legacy custom parsers
            if suffix == ".docx":
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = "\n".join(para.text for para in doc.paragraphs)
                    metadata["paragraphs"] = len(doc.paragraphs)
                except Exception as ex:
                    warnings.append(f"DOCX fallback failure: {ex}")
                    confidence = 0.3
            elif suffix in {".xlsx", ".xls"}:
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path)
                    try:
                        lines = []
                        for sheet in wb.sheetnames:
                            ws = wb[sheet]
                            lines.append(f"Sheet: {sheet}")
                            for row in ws.iter_rows(values_only=True):
                                lines.append("\t".join(str(v) if v else "" for v in row))
                        text = "\n".join(lines)
                        metadata["sheets"] = wb.sheetnames
                    finally:
                        wb.close()
                except Exception as ex:
                    warnings.append(f"XLSX fallback failure: {ex}")
                    confidence = 0.3
            elif suffix == ".pptx":
                try:
                    from pptx import Presentation
                    prs = Presentation(file_path)
                    lines = []
                    for slide_num, slide in enumerate(prs.slides, 1):
                        lines.append(f"Slide {slide_num}:")
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                lines.append(shape.text)
                    text = "\n".join(lines)
                    metadata["slides"] = len(prs.slides)
                except Exception as ex:
                    warnings.append(f"PPTX fallback failure: {ex}")
                    confidence = 0.3

        if not text:
            confidence = 0.0

        return DocumentExtract(
            original_filename=file_path.name,
            document_type="office",
            extracted_text=text,
            metadata=metadata,
            confidence=confidence,
            warnings=warnings,
            extraction_tool=extraction_tool,
        )

    def _extract_image(self, file_path: Path) -> DocumentExtract:
        """Extract from images (OCR)."""
        try:
            import pytesseract
            from PIL import Image

            with Image.open(file_path) as img:
                text = pytesseract.image_to_string(img)
            confidence = 0.6  # OCR confidence varies
            warnings: list[str] = []
        except ImportError:
            text = ""
            confidence = 0.0
            warnings: list[str] = ["pytesseract/Tesseract not installed"]
        except Exception as e:
            logger.warning(f"Image extraction error: {e}")
            text = ""
            confidence = 0.0
            warnings: list[str] = [str(e)]

        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0

        return DocumentExtract(
            original_filename=file_path.name,
            document_type="image",
            extracted_text=text,
            metadata={"file_size_bytes": file_size},
            confidence=confidence,
            warnings=warnings,
            extraction_tool="tesseract_ocr",
        )

    def _extract_audio(self, file_path: Path) -> DocumentExtract:
        """Extract from audio files (Whisper)."""
        try:
            import whisper

            model = whisper.load_model("base")
            result = whisper.transcribe(str(file_path))
            text = result["text"]
            confidence = 0.75
            warnings: list[str] = []
        except ImportError:
            text = ""
            confidence = 0.0
            warnings: list[str] = ["openai-whisper not installed"]
        except Exception as e:
            logger.warning(f"Audio extraction error: {e}")
            text = ""
            confidence = 0.0
            warnings: list[str] = [str(e)]

        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0

        return DocumentExtract(
            original_filename=file_path.name,
            document_type="audio",
            extracted_text=text,
            metadata={"file_size_bytes": file_size},
            confidence=confidence,
            warnings=warnings,
            extraction_tool="whisper",
        )
