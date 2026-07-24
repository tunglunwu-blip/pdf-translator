import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("缺少 PyMuPDF。請執行: pip install pymupdf deep-translator streamlit") from exc

# ==========================================
# 核心邏輯：公式保護、字距清理、PDF排版
# ==========================================

RectTuple = Tuple[float, float, float, float]
RgbColor = Tuple[float, float, float]

FIGURE_REFERENCE_PATTERN = re.compile(
    r"(?P<open>[\(（]\s*)"
    r"(?P<reference>"
    r"(?:圖|Fig(?:ure)?\.?)\s*"
    r"\d+[A-Za-z]?"
    r"(?:\s*[,、，;；/–—~～\-]\s*(?:\d+)?[A-Za-z]?)*"
    r")"
    r"(?P<close>\s*[\)）])",
    flags=re.IGNORECASE,
)

@dataclass
class TextRegion:
    page_number: int
    rect: RectTuple
    source_text: str
    font_size: float
    color: Tuple[float, float, float]
    rotation: int
    alignment: int
    translated_text: str = ""

_CJK_CHAR = (
    r"\u2E80-\u2EFF\u3000-\u303F\u31C0-\u31EF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"
)
_CJK_RE = f"[{_CJK_CHAR}]"
_TECH_ASCII_RE = r"[A-Za-z0-9%°℃µμ]"

def normalize_pdf_text(lines: Sequence[str]) -> str:
    text = "\n".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()

def normalize_translation_spacing(text: str) -> str:
    """自動修復中文黏字，並保留英數字正確空格"""
    cleaned = (text or "").replace("\u00A0", " ").replace("\u3000", " ")
    cleaned = re.sub(r"[ \t\r\n]+", " ", cleaned).strip()
    cleaned = re.sub(rf"(?<={_CJK_RE})\s+(?={_CJK_RE})", "", cleaned)
    cleaned = re.sub(rf"(?<={_CJK_RE})\s+(?={_TECH_ASCII_RE})", "", cleaned)
    cleaned = re.sub(rf"(?<={_TECH_ASCII_RE})\s+(?={_CJK_RE})", "", cleaned)
    cleaned = re.sub(r"([（(《【〔「『])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([）)》】〕」』，。；：！？、])", r"\1", cleaned)
    cleaned = re.sub(rf"(?<={_CJK_RE})\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])\s+(?=[）)》】〕」』])", r"\1", cleaned)
    return cleaned.strip()

def is_translatable(text: str) -> bool:
    """過濾公式、數字列與網址，確保原文書公式不被亂翻"""
    stripped = text.strip()
    if not stripped or len(stripped) < 2:
        return False
    if re.fullmatch(r"(?:https?://|www\.)\S+", stripped, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"(?:doi\s*:\s*)?10\.\d{4,9}/\S+", stripped, flags=re.IGNORECASE):
        return False

    latin_letters = re.findall(r"[A-Za-z]", stripped)
    latin_words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", stripped)
    if len(latin_letters) < 3 or not latin_words:
        return False

    visible = [c for c in stripped if not c.isspace()]
    math_marks = re.findall(r"[=+×÷∫∑√≈≠≤≥^{}]|(?:[_^]\{?)", stripped)
    digit_count = sum(c.isdigit() for c in visible)
    latin_ratio = len(latin_letters) / max(1, len(visible))

    if len(math_marks) >= 2 and latin_ratio < 0.45:
        return False
    if digit_count > len(latin_letters) * 2 and len(latin_words) <= 2:
        return False
    return True

def int_color_to_rgb(value: int) -> Tuple[float, float, float]:
    return (
        ((value >> 16) & 255) / 255.0,
        ((value >> 8) & 255) / 255.0,
        (value & 255) / 255.0,
    )

def line_rotation(direction: Sequence[float]) -> int:
    if len(direction) != 2:
        return 0
    angle = math.degrees(math.atan2(-direction[1], direction[0])) % 360
    candidates = (0, 90, 180, 270)
    nearest = min(candidates, key=lambda x: min(abs(angle - x), 360 - abs(angle - x)))
    return nearest if min(abs(angle - nearest), 360 - abs(angle - nearest)) <= 8 else 0

def infer_alignment(rect: fitz.Rect, page_rect: fitz.Rect) -> int:
    center_error = abs(rect.x0 + rect.x1 - page_rect.x0 - page_rect.x1)
    if rect.width < page_rect.width * 0.8 and center_error < page_rect.width * 0.08:
        return fitz.TEXT_ALIGN_CENTER
    return fitz.TEXT_ALIGN_LEFT

def split_line_spans(line_spans: Sequence[dict]) -> List[List[dict]]:
    if not line_spans:
        return []
    ordered = sorted(line_spans, key=lambda span: float(span["bbox"][0]))
    groups: List[List[dict]] = [[ordered[0]]]
    for span in ordered[1:]:
        previous = groups[-1][-1]
        gap = float(span["bbox"][0]) - float(previous["bbox"][2])
        typical_size = median([float(previous.get("size", 10.0)), float(span.get("size", 10.0))])
        if gap > max(24.0, typical_size * 4.0):
            groups.append([span])
        else:
            groups[-1].append(span)
    return groups

def join_line_spans(line_spans: Sequence[dict]) -> str:
    if not line_spans:
        return ""
    ordered = sorted(line_spans, key=lambda span: float(span["bbox"][0]))
    pieces: List[str] = []
    previous: Optional[dict] = None
    for span in ordered:
        span_text = str(span.get("text", ""))
        if not span_text:
            continue
        if previous is not None and pieces:
            previous_text = str(previous.get("text", ""))
            gap = float(span["bbox"][0]) - float(previous["bbox"][2])
            typical_size = median([float(previous.get("size", 10.0)), float(span.get("size", 10.0))])
            if gap > max(0.8, typical_size * 0.12) and previous_text and not previous_text[-1].isspace() and not span_text[0].isspace():
                pieces.append(" ")
        pieces.append(span_text)
        previous = span
    return "".join(pieces)

def make_region(page: fitz.Page, page_number: int, text: str, spans: Sequence[dict], rect: fitz.Rect, direction: Sequence[float]) -> Optional[TextRegion]:
    if not spans or not is_translatable(text):
        return None
    rect &= page.rect
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return None

    sizes = [float(span.get("size", 10.0)) for span in spans]
    colors = [int(span.get("color", 0)) for span in spans]
    return TextRegion(
        page_number=page_number,
        rect=(rect.x0, rect.y0, rect.x1, rect.y1),
        source_text=text,
        font_size=max(4.0, median(sizes)),
        color=int_color_to_rgb(colors[0] if colors else 0),
        rotation=line_rotation(direction),
        alignment=infer_alignment(rect, page.rect),
    )

def extract_regions(page: fitz.Page, page_number: int) -> List[TextRegion]:
    regions: List[TextRegion] = []
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        line_texts: List[str] = []
        spans: List[dict] = []
        directions: List[Sequence[float]] = []
        split_groups: List[Tuple[List[dict], Sequence[float], fitz.Rect]] = []
        line_group_counts: List[int] = []
        for line in block.get("lines", []):
            line_spans = [span for span in line.get("spans", []) if span.get("text", "")]
            if not line_spans:
                continue
            line_texts.append(join_line_spans(line_spans))
            spans.extend(line_spans)
            direction = line.get("dir", (1.0, 0.0))
            directions.append(direction)
            groups = split_line_spans(line_spans)
            line_group_counts.append(len(groups))
            for group in groups:
                group_rect = fitz.Rect(group[0]["bbox"])
                for span in group[1:]:
                    group_rect |= fitz.Rect(span["bbox"])
                split_groups.append((group, direction, group_rect))

        word_count = sum(len(re.findall(r"[A-Za-z][A-Za-z'-]*", line_text)) for line_text in line_texts)
        looks_like_prose = len(line_texts) >= 2 and word_count >= 8
        block_has_wide_gap = not looks_like_prose and any(group_count > 1 for group_count in line_group_counts)

        if block_has_wide_gap:
            for group, direction, group_rect in split_groups:
                group_text = normalize_pdf_text([join_line_spans(group)])
                region = make_region(page, page_number, group_text, group, group_rect, direction)
                if region is not None:
                    regions.append(region)
            continue

        text = normalize_pdf_text(line_texts)
        rect = fitz.Rect(block["bbox"])
        region = make_region(page, page_number, text, spans, rect, directions[0] if directions else (1.0, 0.0))
        if region is not None:
            regions.append(region)
    return regions

class TranslationCache:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: Dict[str, str] = {}
        if enabled and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = {str(k): str(v) for k, v in loaded.items()}
            except (OSError, json.JSONDecodeError):
                pass

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(("google|en|zh-TW|" + text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[str]:
        return self.data.get(self.key(text)) if self.enabled else None

    def put(self, text: str, translation: str) -> None:
        if self.enabled:
            self.data[self.key(text)] = translation

    def save(self) -> None:
        if not self.enabled:
            return
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

def translate_regions(regions: Sequence[TextRegion], cache: TranslationCache, delay: float, workers: int, status_callback=None) -> None:
    from deep_translator import GoogleTranslator

    unique: Dict[str, str] = {}
    for region in regions:
        cached = cache.get(region.source_text)
        if cached:
            region.translated_text = normalize_translation_spacing(cached)
        else:
            unique.setdefault(cache.key(region.source_text), region.source_text)

    pending = list(unique.items())
    if not pending:
        return

    translated_by_key: Dict[str, str] = {}
    thread_state = threading.local()

    def translate_one(key: str, source: str) -> Tuple[str, str, str]:
        translator = getattr(thread_state, "translator", None)
        if translator is None:
            translator = GoogleTranslator(source="en", target="zh-TW")
            thread_state.translator = translator

        last_error = None
        for attempt in range(1, 4):
            try:
                translated = translator.translate(source)
                cleaned = normalize_translation_spacing((translated or "").strip())
                if not cleaned:
                    cleaned = source
                if delay > 0:
                    time.sleep(delay)
                return key, source, cleaned
            except Exception as exc:
                last_error = exc
                time.sleep(attempt * 1.5)

        return key, source, source  # 重試失敗則退回原文

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(translate_one, key, source): (key, source) for key, source in pending}
        for future in as_completed(futures):
            key, source, clean_translation = future.result()
            translated_by_key[key] = clean_translation
            cache.put(source, clean_translation)
            completed += 1
            if status_callback and completed % 5 == 0:
                status_callback(completed, len(pending))

    cache.save()
    for region in regions:
        if not region.translated_text:
            translated = translated_by_key.get(cache.key(region.source_text), cache.get(region.source_text) or "")
            region.translated_text = normalize_translation_spacing(translated)

def resolve_chinese_font() -> Optional[Path]:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windows_dir / "Fonts" / "msjh.ttc",
        windows_dir / "Fonts" / "msjhbd.ttc",
        windows_dir / "Fonts" / "mingliu.ttc",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    return next((path for path in candidates if path.is_file()), None)

def prepare_font(page: fitz.Page, font_path: Optional[Path]) -> str:
    if font_path is None:
        return "china-t"
    font_name = "CustomZhFont"
    page.insert_font(fontname=font_name, fontfile=str(font_path))
    return font_name

def redact_original_text(page: fitz.Page, regions: Sequence[TextRegion]) -> None:
    for region in regions:
        rect = fitz.Rect(region.rect)
        page.add_redact_annot(rect, fill=None, cross_out=False)
    if regions:
        try:
            page.apply_redactions(images=0, graphics=0, text=0)
        except TypeError:
            page.apply_redactions(images=0)

def insert_fitted_text(page: fitz.Page, region: TextRegion, font_name: str, font_path: Optional[Path], font_scale: float, min_font_size: float, font_boldness: float) -> bool:
    rect = fitz.Rect(region.rect)
    size = max(min_font_size, region.font_size * font_scale)
    render_mode = 2 if font_boldness > 0 else 0

    while size >= min_font_size - 1e-6:
        remaining = page.insert_textbox(
            rect,
            region.translated_text,
            fontname=font_name,
            fontfile=str(font_path) if font_path is not None else None,
            fontsize=size,
            color=region.color,
            fill=region.color,
            align=region.alignment,
            rotate=region.rotation,
            lineheight=1.08,
            render_mode=render_mode,
            border_width=font_boldness,
            overlay=True,
        )
        if remaining >= 0:
            return True
        size -= 0.4
    return False

# ==========================================
# Streamlit 網頁 UI 介面
# ==========================================

st.set_page_config(
    page_title="PDF 原文書專業翻譯器",
    page_icon="📚",
    layout="centered"
)

st.title("📚 PDF 原文書專業翻譯器")
st.markdown("##### 免費 · 排版保持 · 字不黏在一起 · 公式自動保護")

with st.sidebar:
    st.header("⚙️ 翻譯設定")
    workers = st.slider("併行工作數 (Workers)", min_value=1, max_value=5, value=3, help="網路不穩定或被擋時改為 1")
    delay = st.slider("請求延遲 (秒)", min_value=0.0, max_value=2.0, value=0.20, step=0.05)
    font_scale = st.slider("中文字縮放比", min_value=0.5, max_value=1.0, value=0.92)
    min_font_size = st.number_input("最小字體 (pt)", min_value=2.0, max_value=10.0, value=4.0)

uploaded_file = st.file_uploader("選擇要翻譯的英文 PDF 檔案", type=["pdf"])

if uploaded_file is not None:
    temp_dir = Path("temp_pdf")
    temp_dir.mkdir(exist_ok=True)
    
    input_pdf_path = temp_dir / uploaded_file.name
    output_pdf_path = temp_dir / f"{input_pdf_path.stem}_繁中.pdf"
    
    with open(input_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"已上傳檔案：**{uploaded_file.name}**")
    
    if st.button("🚀 開始翻譯全書", type="primary"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_box.info("🔍 正在分析 PDF 頁面結構...")
            doc = fitz.open(str(input_pdf_path))
            total_pages = doc.page_count
            
            all_regions = []
            page_regions = {}
            for page_index in range(total_pages):
                page = doc[page_index]
                regions = extract_regions(page, page_index)
                page_regions[page_index] = regions
                all_regions.extend(regions)
                
            status_box.info(f"📄 全書共 {total_pages} 頁，找到 {len(all_regions)} 個文字段落。開始連線翻譯...")
            
            cache_path = output_pdf_path.with_name(f".{output_pdf_path.stem}_翻譯快取.json")
            cache = TranslationCache(cache_path, enabled=True)
            
            def update_status(done, total):
                status_box.info(f"🌐 正在翻譯中... ({done}/{total} 段落)")
                
            translate_regions(all_regions, cache, delay=delay, workers=workers, status_callback=update_status)
            
            status_box.info("🎨 翻譯完成，正在繪製中文頁面排版...")
            chinese_font = resolve_chinese_font()
            
            for page_index in range(total_pages):
                regions = [r for r in page_regions.get(page_index, []) if r.translated_text]
                if regions:
                    page = doc[page_index]
                    font_name = prepare_font(page, chinese_font)
                    redact_original_text(page, regions)
                    for region in regions:
                        insert_fitted_text(page, region, font_name, chinese_font, font_scale, min_font_size, 0.12)
                progress_bar.progress((page_index + 1) / total_pages)
                
            doc.save(str(output_pdf_path), garbage=4, deflate=True)
            doc.close()
            
            status_box.success("🎉 翻譯完成！")
            
            with open(output_pdf_path, "rb") as f:
                st.download_button(
                    label="📥 點此下載翻譯好的中文 PDF",
                    data=f,
                    file_name=output_pdf_path.name,
                    mime="application/pdf",
                    type="primary"
                )
                
        except Exception as e:
            st.error(f"❌ 執行出錯：{str(e)}")