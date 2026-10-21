import os
import re
import sqlite3

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from datetime import datetime

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
    convert_from_bytes = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

from logger import logger


class TokenStorage:
    """Хранилище уникальных токенов текста для последующего обучения."""

    def __init__(self, db_path="results/unique_tokens.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unique_tokens (
                    token TEXT PRIMARY KEY,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    source_file TEXT,
                    seen_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT,
                    block_hash TEXT UNIQUE,
                    block_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_unique_tokens_seen_count ON unique_tokens(seen_count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_blocks_source_file ON processed_blocks(source_file)"
            )

    @staticmethod
    def _normalize_token(token):
        if token is None:
            return ""
        return str(token).replace(" ", " ").strip()

    @staticmethod
    def _tokenize_block_text(text):
        if text is None:
            return []
        normalized = str(text).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        return [normalized]

    def store_block_tokens(self, block_text, source_file=""):
        if not block_text or not str(block_text).strip():
            return 0

        tokens = self._tokenize_block_text(block_text)
        if not tokens:
            return 0

        block_hash = self._hash_block(block_text)
        now = datetime.utcnow().isoformat(timespec="seconds")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO processed_blocks (source_file, block_hash, block_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (source_file, block_hash, str(block_text), now),
            )

            for raw_token in tokens:
                clean = self._normalize_token(raw_token)
                if not clean:
                    continue
                conn.execute(
                    """
                    INSERT INTO unique_tokens (token, first_seen_at, last_seen_at, source_file, seen_count)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(token) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        source_file = excluded.source_file,
                        seen_count = unique_tokens.seen_count + 1
                    """,
                    (clean, now, now, source_file),
                )

        return len(set(tokens))

    @staticmethod
    def _hash_block(block_text):
        text = "" if block_text is None else str(block_text)
        return str(abs(hash(text.strip())))

    def get_unique_tokens(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT token FROM unique_tokens ORDER BY token").fetchall()
            return [row["token"] for row in rows]

    def get_token_stats(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT token, seen_count, source_file, last_seen_at FROM unique_tokens ORDER BY token"
            ).fetchall()
            return [dict(row) for row in rows]


class BasePdfProcessor(ABC):
    def __init__(self, token_storage=None):
        self.token_storage = token_storage

    @classmethod
    @abstractmethod
    def supports(cls, file_name):
        raise NotImplementedError

    @abstractmethod
    def process(self, pdf_bytes, source_name=""):
        raise NotImplementedError

    def _store_block_text(self, block_text, source_name=""):
        if self.token_storage is not None:
            self.token_storage.store_block_tokens(block_text, source_name)


class OCRPdfProcessor(BasePdfProcessor):
    """OCR-процессор для .dwg.pdf."""

    @classmethod
    def supports(cls, file_name):
        name = str(file_name or "").lower()
        return ".dwg.pdf" in name or name.endswith(".dwg.pdf")

    def process(self, pdf_bytes, source_name=""):
        if cv2 is None or np is None or pytesseract is None or convert_from_bytes is None:
            raise RuntimeError("Для OCR требуется cv2 / numpy / pytesseract / pdf2image")

        logger.info(f"OCRPdfProcessor: обработка файла {source_name or 'без имени'}")
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        if not pages:
            raise RuntimeError("Не удалось конвертировать PDF в изображения")

        image = np.array(pages[0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        custom_config = r"--psm 6 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(binary, lang="rus", config=custom_config).strip()
        cleaned = self._clean_ocr_text(text)
        if cleaned:
            logger.info(
                "OCRPdfProcessor: пропускаем запись OCR-результата в TokenStorage для %s; dwg.pdf должен уходить в API без записи в БД.",
                source_name or 'без имени',
            )
        return {
            "text": cleaned,
            "blocks": [{"text": cleaned, "source": source_name}],
            "blocks_data": {"block_1": {"text": cleaned, "source": source_name}},
        }

    @staticmethod
    def _clean_ocr_text(text):
        if not text:
            return ""
        filtered = re.sub(r"\r+", "\n", text)
        filtered = filtered.replace("\xa0", " ")
        filtered = re.sub(r"\n{3,}", "\n\n", filtered)
        return filtered.strip()


class TextLayerPdfProcessor(BasePdfProcessor):
    """Text-layer extractor для .cdw/.cwd PDF, с базовой фильтрацией технических требований."""

    STRONG_KEYWORDS = (
        "ГОСТ",
        "ОСТ",
        "ТУ",
        "РД",
        "МАТЕРИАЛ",
        "ПОКРЫТИЕ",
        "РАЗМЕР",
        "РАДИУС",
        "СВАРКА",
        "КОНТРОЛЬ",
        "ДОПУСК",
        "ИНСТРУМЕНТ",
        "ПРИМЕЧАНИЕ",
    )

    @classmethod
    def supports(cls, file_name):
        name = str(file_name or "").lower()
        return (
            ".cdw.pdf" in name or name.endswith(".cdw.pdf") or
            ".cwd.pdf" in name or name.endswith(".cwd.pdf") or
            ".cdw" in name or ".cwd" in name
        )

    @staticmethod
    def _normalize_block_text(value):
        text = str(value or "").replace("\r", "\n")
        text = text.replace("\xa0", " ")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()

    @staticmethod
    def _is_noise_line(line):
        text = re.sub(r"\s+", " ", str(line or "")).strip()
        if not text:
            return True
        upper = text.upper()
        if len(text) < 3:
            return True
        if "СТЕКЛОТЕКСТОЛИТ" in upper and "МАРКИ" in upper:
            return True
        if upper.startswith("СТЕКЛОТЕКСТОЛИТ"):
            return True
        if any(token in upper for token in (
            "ДОКУМЕНТ ПОДПИСАН",
            "СЕРТИФИКАТ",
            "ЭЦП",
            "ВЛАДЕЛЕЦ ПОДПИСИ",
            "ДЕЙСТВИТЕЛЬНО С",
            "КОПИРОВАЛ",
            "ФОРМАТ",
            "CRC",
            "ID ВЕРСИИ",
            "ИНФОРМАЦИЯ",
            "ПОДТ.",
            "РАЗРАБ.",
            "ПРОВ.",
            "Н.КОНТР.",
            "Т.КОНТР.",
            "ИЗМ.",
            "ЛИСТ",
            "ПОДП.",
            "ДАТА",
        )):
            return True
        if upper.startswith("A4") or upper.startswith("A3") or upper.startswith("A5"):
            return True
        if re.fullmatch(r"[0-9A-F]{8,}", upper):
            return True
        if re.fullmatch(r"[\W_]+", text):
            return True
        if re.fullmatch(r"[-=|/\\]+", text):
            return True
        if re.fullmatch(r"[A-ZА-ЯЁ]{1,3}", text):
            return True
        return False

    @staticmethod
    def _filter_requirement_lines(text):
        raw_lines = str(text or "").replace("\r", "\n").split("\n")
        kept = []
        for raw in raw_lines:
            candidate = re.sub(r"\s+", " ", raw).strip()
            if not candidate:
                continue
            upper = candidate.upper()
            if upper.startswith("СТЕКЛОТЕКСТОЛИТ") or ("СТЕКЛОТЕКСТОЛИТ" in upper and "МАРКИ" in upper):
                continue
            if TextLayerPdfProcessor._is_noise_line(candidate):
                continue
            keyword_patterns = (
                "ГОСТ", "ОСТ", "ТУ", "МАТЕРИАЛ", "ПОКРЫТИЕ", "РАДИУС", "ДИАМЕТР",
                "ШЕРОХОВАТОСТЬ", "СВАРКА", "РЕЗЬБА", "ПРИМЕЧАНИЕ", "ФОСФОР",
                "МЕДЬ", "ДОЛЯ", "НЕ БОЛЕЕ", "НЕ МЕНЕЕ", "МАРКИРОВАТЬ", "МАССОВОЙ",
                "МАССОВАЯ", "ПОСТАВЛЯТЬСЯ", "ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
                "ДОПУСКАЕТСЯ", "СКОБКАХ", "ШРИФТОМ", "СБОРКЕ", "СБОРКИ"
            )
            has_requirement_keyword = any(re.search(rf"\b{re.escape(keyword)}\b", upper) is not None for keyword in keyword_patterns)
            has_numeric = any(ch.isdigit() for ch in candidate)
            has_letters = any(ch.isalpha() for ch in candidate)

            numbered_match = re.match(r"^(?:[-*•\s]*)(?P<num>\d{1,2})\s*(?:[\.\)\-]|[*•])?\s*(?P<body>.+)$", candidate)
            if numbered_match:
                body = (numbered_match.group("body") or "").strip()
                if not body:
                    continue
                short_numbered_noise = (
                    len(body) <= 18
                    and not has_requirement_keyword
                    and not re.search(r"[A-Za-zА-ЯЁ]{4,}", body)
                )
                if short_numbered_noise:
                    continue
                kept.append(candidate)
                continue

            last_kept = kept[-1] if kept else ""
            previous_numbered = bool(re.match(r"^\d", last_kept))
            reference_prefixes = ("ГОСТ ", "ОСТ ", "ТУ ", "СТАЛЬ ", "МАТЕРИАЛ ", "CRC", "A4", "A3", "A5", "ID ВЕРСИИ")
            reference_like = candidate.upper().startswith(reference_prefixes)
            continuation = (
                bool(kept)
                and previous_numbered
                and not re.match(r"^\d", candidate)
                and not reference_like
                and not TextLayerPdfProcessor._is_noise_line(candidate)
            )
            if continuation:
                kept.append(candidate)
                continue
            if re.match(r"^(?:СТАЛЬ|МАТЕРИАЛ)\b", candidate, flags=re.IGNORECASE):
                continue

            if not (has_letters and has_numeric):
                continue

            strong_requirement_markers = (
                "НЕ БОЛЕЕ", "НЕ МЕНЕЕ", "ДОПУСКАЕТСЯ", "МАРКИРОВАТЬ",
                "РАЗМЕРЫ", "РАЗМЕР", "РАДИУС", "ДОПУСКИ", "ДОПУСК",
                "ПРИМЕЧАНИЕ", "ПОСТАВЛЯТЬСЯ", "ЗАМЕНИТЕЛЬ", "СБОРКЕ",
                "СБОРКИ", "ОБЕСПЕЧ", "ОБЩИЕ", "МАТЕРИАЛ-ЗАМЕНИТЕЛЬ"
            )
            has_strong_requirement_marker = any(marker in upper for marker in strong_requirement_markers)
            if has_requirement_keyword and not has_strong_requirement_marker and not continuation:
                continue

            if has_requirement_keyword or len(candidate) >= 25:
                kept.append(candidate)

        return "\n".join(kept)

    @staticmethod
    def _looks_like_requirement_block(text):
        if not text:
            return False
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(text)
        return bool(cleaned)
    @staticmethod
    def _looks_like_requirement_block(text):
        if not text:
            return False
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(text)
        return bool(cleaned)

    def process(self, pdf_bytes, source_name=''):
        if fitz is None:
            raise RuntimeError('PyMuPDF (fitz) не установлен. Невозможно использовать текстовый слой.')

        logger.info(f"TextLayerPdfProcessor: обработка файла {source_name or 'без имени'}")
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        blocks = []
        seen = set()

        try:
            for page in doc:
                for block in page.get_text('blocks'):
                    text = self._normalize_block_text(block[4])
                    if not text:
                        continue
                    cleaned_block = self._filter_requirement_lines(text)
                    if not cleaned_block:
                        continue
                    normalized_block = re.sub(r"[ \t]+\n", "\n", cleaned_block)
                    normalized_block = re.sub(r"\n{3,}", "\n\n", normalized_block).strip()
                    if not normalized_block:
                        continue
                    block_key = normalized_block.lower()
                    if block_key in seen:
                        continue
                    seen.add(block_key)
                    blocks.append(normalized_block)
                    self._store_block_text(normalized_block, source_name)
        finally:
            doc.close()

        full_text = "\n\n".join(blocks)
        return {
            'text': full_text,
            'blocks': blocks,
            'blocks_data': {f'block_{idx + 1}': {'text': block, 'source': source_name} for idx, block in enumerate(blocks)},
        }


class PdfProcessingFactory:
    """Фабрика процессоров для выбора реализации по имени файла."""

    @staticmethod
    def _reload_env():
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)

    @staticmethod
    def _env_flag(name, default=True):
        PdfProcessingFactory._reload_env()
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return str(raw_value).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}

    @staticmethod
    def _normalize_type_token(value):
        token = str(value or "").strip().lower().replace("\\", "/")
        token = token.split("#", 1)[0].strip()
        token = token.replace(" ", "").replace("_", "")
        token = token.strip(".")
        return token
    @classmethod
    def _allowed_file_types(cls):
        cls._reload_env()
        allowed = set()
        for env_name in ("ALLOWED_FILE_TYPES", "ALLOWED_PDF_TYPES", "FILE_TYPES_ALLOWED"):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            for item in re.split(r"[,;\s]+", str(raw_value)):
                token = cls._normalize_type_token(item)
                if token:
                    allowed.add(token)
        return allowed

    @classmethod
    def _disabled_file_types(cls):
        cls._reload_env()
        disabled = set()
        for env_name in ("DISABLED_FILE_TYPES", "DISABLED_PDF_TYPES", "DISABLED_TYPES", "FILE_TYPES_DISABLED"):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            for item in re.split(r"[,;\s]+", str(raw_value)):
                token = cls._normalize_type_token(item)
                if token:
                    disabled.add(token)
        return disabled

    @classmethod
    def _file_type_tokens(cls, file_name):
        normalized = str(file_name or "").lower()
        tokens = set()
        for suffix in (".dwg.pdf", ".cdw.pdf", ".cwd.pdf", ".dwg", ".cdw", ".cwd"):
            if suffix in normalized:
                tokens.update({
                    cls._normalize_type_token(suffix),
                    cls._normalize_type_token(suffix.lstrip(".")),
                    cls._normalize_type_token(suffix.replace(".pdf", "")),
                })
                break
        if not tokens:
            return tokens
        if ".dwg.pdf" in normalized or normalized.endswith(".dwg.pdf") or ".dwg" in normalized:
            tokens.update({"dwg", "dwg.pdf"})
        if ".cdw.pdf" in normalized or normalized.endswith(".cdw.pdf") or ".cdw" in normalized:
            tokens.update({"cdw", "cdw.pdf"})
        if ".cwd.pdf" in normalized or normalized.endswith(".cwd.pdf") or ".cwd" in normalized:
            tokens.update({"cwd", "cwd.pdf"})
        return tokens

    @classmethod
    def is_processing_enabled(cls, file_name):
        cls._reload_env()
        normalized = str(file_name or "").lower()
        file_type_tokens = cls._file_type_tokens(normalized)
        allowed_types = cls._allowed_file_types()
        if allowed_types:
            if not file_type_tokens:
                return False
            return bool(file_type_tokens & allowed_types)

        disabled_types = cls._disabled_file_types()
        if file_type_tokens and (file_type_tokens & disabled_types):
            return False

        if ".dwg.pdf" in normalized or normalized.endswith(".dwg.pdf") or ".dwg" in normalized:
            return cls._env_flag("ENABLE_DWG_PDF_PROCESSING", True)
        if ".cdw.pdf" in normalized or normalized.endswith(".cdw.pdf") or ".cdw" in normalized:
            return cls._env_flag("ENABLE_CDW_PDF_PROCESSING", True)
        if ".cwd.pdf" in normalized or normalized.endswith(".cwd.pdf") or ".cwd" in normalized:
            return cls._env_flag("ENABLE_CWD_PDF_PROCESSING", True)
        return cls._env_flag("ENABLE_UNKNOWN_PDF_PROCESSING", True)

    @staticmethod
    def create(file_name, token_storage=None):
        normalized = str(file_name or '').lower()
        if OCRPdfProcessor.supports(normalized):
            if not PdfProcessingFactory.is_processing_enabled(normalized):
                return None
            return OCRPdfProcessor(token_storage=token_storage)
        if TextLayerPdfProcessor.supports(normalized):
            if not PdfProcessingFactory.is_processing_enabled(normalized):
                return None
            return TextLayerPdfProcessor(token_storage=token_storage)
        if fitz is not None:
            if not PdfProcessingFactory.is_processing_enabled(normalized):
                return None
            return TextLayerPdfProcessor(token_storage=token_storage)
        if not PdfProcessingFactory.is_processing_enabled(normalized):
            return None
        return OCRPdfProcessor(token_storage=token_storage)

__all__ = [
    'BasePdfProcessor',
    'OCRPdfProcessor',
    'TextLayerPdfProcessor',
    'PdfProcessingFactory',
    'TokenStorage',
]
