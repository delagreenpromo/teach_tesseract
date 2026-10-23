import os
import sys
import datetime
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_bytes
from logger import logger
import re

os.environ["QT_QPA_PLATFORM"] = "xcb"


class OCRProcessor:
    _STAMP_MARKERS = (
        "документ подписан",
        "сертификат :",
        "сертификат:",
        "владелец подписи",
        "действительно с",
        "копировал",
        "копиробал",
        "версия:",
        "эцп",
        "лит.",
        "масса",
        "масшт",
        "сборочный чертеж",
        "формат а4",
        "формат а3",
        "формат а2",
        "формат а1",
        "формат а0",
    )

    def __init__(self, dpi=300, lang="rus", psm=6, char_whitelist="", min_area_ratio=0.0005, width_tolerance=0.3,
                 save_images=False, save_dir=None, use_advanced=True,
                 kernel_width=40, kernel_height=60, dilation_iterations=2,
                 debug=False, production_mode=True, frac_height_ratio=2.0, word_merge_kernel_width=15):
        self.dpi = dpi
        self.lang = lang
        self.psm = psm
        self.min_area_ratio = min_area_ratio
        self.width_tolerance = width_tolerance
        self.save_images = save_images
        self.save_dir = save_dir or "ocr_debug"
        self.use_advanced = use_advanced
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.dilation_iterations = dilation_iterations
        self.debug = debug
        self.production_mode = production_mode
        self.char_whitelist = char_whitelist
        self.frac_height_ratio = frac_height_ratio
        self.word_merge_kernel_width = word_merge_kernel_width

        if self.save_images or not self.production_mode:
            os.makedirs(self.save_dir, exist_ok=True)

    def pdf_to_images(self, pdf_bytes):
        try:
            pil_images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
            images = [np.array(img) for img in pil_images]
            logger.debug(f"PDF преобразован в {len(images)} изображений")
            return images
        except Exception as e:
            logger.error(f"Ошибка конвертации PDF: {e}")
            return []

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        return binary

    def _is_real_text_block(self, roi):
        if roi.size == 0:
            return False
        try:
            data = pytesseract.image_to_data(
                roi, lang=self.lang, output_type=Output.DICT, config='--psm 6'
            )
        except Exception:
            return False

        good_words = 0
        total_conf = 0
        n_conf = 0

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            try:
                conf = int(float(data['conf'][i]))
            except (ValueError, TypeError):
                continue
            if conf < 0:
                continue

            total_conf += conf
            n_conf += 1

            stripped = text.rstrip('.,:;!?')

            if re.fullmatch(r'[А-Яа-яЁёA-Za-z]{3,}', stripped) and conf >= 60:
                good_words += 1
                continue

            if (len(stripped) >= 4
                    and re.fullmatch(r'[А-Яа-яЁёA-Za-z0-9\-./,№()]+', stripped)
                    and re.search(r'[А-Яа-яЁёA-Za-z]{2,}', stripped)
                    and conf >= 50):
                good_words += 1

        avg_conf = (total_conf / n_conf) if n_conf else 0
        return good_words >= 2 or (good_words >= 1 and avg_conf >= 75)

    def is_likely_drawing(self, roi, block_info=None):
        if roi.size == 0:
            return False
        h, w = roi.shape[:2]
        area_total = h * w
        if area_total == 0:
            return False

        if self._is_real_text_block(roi):
            if self.debug and block_info:
                print(f"  Блок {block_info} оставлен как текст (conf-based)")
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contour_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
        contour_ratio = large_contour_area / area_total

        min_line_len = max(30, int(min(w, h) * 0.5))
        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=30,
                                minLineLength=min_line_len, maxLineGap=3)
        line_mask = np.zeros_like(binary)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < min_line_len:
                    continue
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        hough_ratio = cv2.countNonZero(line_mask) / area_total

        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        corner_ratio = len(corners) / area_total

        is_small_block = h < 150 or w < 500
        reasons = []

        if is_small_block:
            if contour_ratio > 0.55 and hough_ratio > 0.10:
                reasons.append(f"small:contour+hough ({contour_ratio:.3f}, {hough_ratio:.3f})")
            if corner_ratio > 0.20 and hough_ratio > 0.15:
                reasons.append(f"small:corner+hough ({corner_ratio:.3f}, {hough_ratio:.3f})")
        else:
            if contour_ratio > 0.3:
                reasons.append(f"contour_ratio={contour_ratio:.3f}")
            if hough_ratio > 0.02 and (contour_ratio > 0.05 or corner_ratio > 0.05):
                reasons.append(f"hough_ratio={hough_ratio:.3f}")
            if corner_ratio > 0.1:
                reasons.append(f"corner_ratio={corner_ratio:.3f}")
            if contour_ratio > 0.15 and hough_ratio > 0.01:
                reasons.append(f"contour+hough ({contour_ratio:.3f}, {hough_ratio:.3f})")

        if reasons:
            if self.debug and block_info:
                print(f"  Блок {block_info} отброшен как чертёж: {', '.join(reasons)}")
            return True
        return False

    def _is_stamp_block(self, text):
        if not text:
            return False
        low = text.lower()
        return any(m in low for m in self._STAMP_MARKERS)

    def _find_fraction_line(self, word_roi):
        gray = cv2.cvtColor(word_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (word_roi.shape[1] // 3, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best_cnt)
        if w < word_roi.shape[1] * 0.3:
            return None
        return y + h // 2

    def filter_drawing_from_roi(self, roi):
        if roi.size == 0:
            return roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        drawing_mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)

            is_line = (aspect_ratio > 12) and (max(w, h) > 60) and (area < 300)
            is_possible_drawing = (aspect_ratio > 8) and (max(w, h) > 80) and (area < 800)

            if is_line or is_possible_drawing:
                roi_fragment = roi[y:y+h, x:x+w]
                if self._is_text_fragment(roi_fragment):
                    continue
                drawing_mask[labels == i] = 255

        cleaned = roi.copy()
        cleaned[drawing_mask > 0] = (255, 255, 255)
        return cleaned

    def _is_text_fragment(self, fragment):
        if fragment.size == 0:
            return False
        try:
            config = '--psm 7 -c tessedit_char_whitelist=0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя.,:;()"'
            res = pytesseract.image_to_string(fragment, lang=self.lang, config=config).strip()
            return len(res) > 0
        except Exception:
            return False

    def handle_fractions_in_block(self, block_image):
        gray = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=30,
                                minLineLength=block_image.shape[1] * 0.4,
                                maxLineGap=3)
        if lines is None:
            return block_image, None

        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > 5:
                continue
            line_ys.append((min(y1, y2) + max(y1, y2)) // 2)

        if not line_ys:
            return block_image, None

        y_center = int(np.median(line_ys))

        top_part = block_image[0:y_center, :]
        bot_part = block_image[y_center:, :]

        top_text = pytesseract.image_to_string(top_part, lang=self.lang, config='--psm 7').strip()
        bot_text = pytesseract.image_to_string(bot_part, lang=self.lang, config='--psm 7').strip()

        if top_text and bot_text:
            fraction_str = f"{top_text}/{bot_text}"
            cleaned = block_image.copy()
            margin = 2
            cleaned[max(0, y_center-margin):min(block_image.shape[0], y_center+margin), :] = (255, 255, 255)
            return cleaned, fraction_str

        return block_image, None

    def get_merged_word_boxes(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, lang=self.lang,
                                         output_type=Output.DICT,
                                         config='--psm 6')
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for i in range(len(data['level'])):
            text = data['text'][i].strip()
            if not text:
                continue
            w, h = data['width'][i], data['height'][i]
            if w <= 0 or h <= 0:
                continue
            x, y = data['left'][i], data['top'][i]
            mask[y:y+h, x:x+w] = 255

        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                 (self.word_merge_kernel_width, 1))
        dilated = cv2.dilate(mask, merge_kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        words = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue
            words.append({'x': x, 'y': y, 'w': w, 'h': h})
        return words

    def analyze_block_for_fractions(self, block_image, frac_height_ratio=None,
                                    min_text_len=2, debug_words=False, debug_image_path=None):
        if frac_height_ratio is None:
            frac_height_ratio = self.frac_height_ratio

        words = self.get_merged_word_boxes(block_image)
        if not words:
            return block_image, [], []

        heights = [wd['h'] for wd in words]
        median_h = np.median(heights)
        if median_h < 5:
            return block_image, [], []

        high_words = [wd for wd in words if wd['h'] >= median_h * frac_height_ratio]

        if debug_words and debug_image_path:
            debug_img = block_image.copy()
            for wd in words:
                color = (0, 255, 0) if wd['h'] < median_h * frac_height_ratio else (0, 0, 255)
                cv2.rectangle(debug_img, (wd['x'], wd['y']),
                              (wd['x']+wd['w'], wd['y']+wd['h']), color, 1)
                cv2.putText(debug_img, f"{wd['h']}", (wd['x'], wd['y']-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(debug_image_path, debug_img)
            print(f"[DEBUG] Слова блока сохранены: {debug_image_path}")

        fraction_texts = []
        high_word_boxes = []
        if high_words:
            cleaned_img = block_image.copy()
            for wd in high_words:
                x, y, w, h = wd['x'], wd['y'], wd['w'], wd['h']
                x1 = max(0, x - 2)
                y1 = max(0, y - 2)
                x2 = min(block_image.shape[1], x + w + 2)
                y2 = min(block_image.shape[0], y + h + 2)
                word_roi = block_image[y1:y2, x1:x2]

                line_y = self._find_fraction_line(word_roi)
                if line_y is None:
                    line_y = word_roi.shape[0] // 2

                top_part = word_roi[0:line_y, :]
                bot_part = word_roi[line_y:, :]

                num_text = pytesseract.image_to_string(top_part, lang=self.lang, config='--psm 7').strip()
                den_text = pytesseract.image_to_string(bot_part, lang=self.lang, config='--psm 7').strip()
                num_clean = re.sub(r'[^\w.,\-]', '', num_text)
                den_clean = re.sub(r'[^\w.,\-]', '', den_text)

                if num_clean and den_clean:
                    fraction_texts.append(f"{num_clean}/{den_clean}")
                    cleaned_img[y1:y2, x1:x2] = (255, 255, 255)
                    high_word_boxes.append((wd['x'], wd['y'], wd['w'], wd['h']))

            return cleaned_img, fraction_texts, high_word_boxes

        return block_image, [], []

    def detect_text_blocks(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        binary = self.preprocess_image(img_rgb)

        data = pytesseract.image_to_data(img_rgb, lang=self.lang, output_type=Output.DICT,
                                         config=f'--psm {self.psm}')

        mask = np.zeros_like(binary)
        for i in range(len(data["level"])):
            if data["text"][i].strip():
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                mask[y:y+h, x:x+w] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_width, self.kernel_height))
        dilated = cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_area = image.shape[0] * image.shape[1] * self.min_area_ratio
        logger.debug(f"min_area = {min_area:.0f} (изображение {image.shape[1]}x{image.shape[0]})")
        text_blocks = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < min_area:
                logger.debug(f"  Блок #{idx} ({w}x{h}) отброшен: площадь {area:.0f} < {min_area:.0f}")
                continue
            roi = image[y:y+h, x:x+w]
            if self.use_advanced and self.is_likely_drawing(roi, block_info=f"#{idx} ({w}x{h})"):
                continue
            text_blocks.append({
                "bbox": (x, y, w, h),
                "area": area,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 1,
            })
        if self.debug:
            print(f"Найдено {len(text_blocks)} текстовых блоков из {len(contours)} контуров")
        return text_blocks

    def sort_blocks_by_columns(self, text_blocks, x_tolerance=None):
        """
        Сортирует блоки в порядке чтения по правилу:
          1. Сначала по X по УБЫВАНИЮ (справа налево).
          2. Блоки с близкими X объединяются в одну «колонку» (x_tolerance).
          3. Внутри колонки — по Y по ВОЗРАСТАНИЮ (сверху вниз).

        Пример:
          [2500,40], [2000,40], [1500,38], [1499,200]
          → [2500,40], [2000,40], [1500,38], [1499,200]

        [2500,40] и [2000,40] — разные X-колонки (X-desc: 2500 → 2000).
        [1500,38] и [1499,200] — одна X-колонка (|1500-1499| ≤ tolerance),
        внутри неё Y-asc: 38 → 200.
        """
        if not text_blocks:
            return []

        if x_tolerance is None:
            x_tolerance = 100

        # Шаг 1: сортируем по X по убыванию (справа налево)
        sorted_by_x = sorted(text_blocks, key=lambda b: b['bbox'][0], reverse=True)

        # Шаг 2: группируем в колонки по X (сравниваем с первым блоком колонки)
        columns = []
        current_column = [sorted_by_x[0]]
        for block in sorted_by_x[1:]:
            column_x = current_column[0]['bbox'][0]
            if abs(block['bbox'][0] - column_x) <= x_tolerance:
                current_column.append(block)
            else:
                # Внутри колонки сортируем по Y по возрастанию
                current_column.sort(key=lambda b: b['bbox'][1])
                columns.append(current_column)
                current_column = [block]
        current_column.sort(key=lambda b: b['bbox'][1])
        columns.append(current_column)

        # Шаг 3: разворачиваем колонки в плоский список
        result = []
        for col in columns:
            result.extend(col)
        return result

    def find_main_blocks_by_width(self, text_blocks, width_tolerance=None):
        """Оставлен для обратной совместимости. В основном потоке не используется."""
        if not text_blocks:
            return [], None
        if width_tolerance is None:
            width_tolerance = self.width_tolerance

        widths = [b['width'] for b in text_blocks]
        median_width = float(np.median(widths))
        min_width = median_width * (1 - width_tolerance)

        matching_blocks = [b for b in text_blocks if b['width'] >= min_width]

        if self.debug:
            print(f"Медианная ширина блоков: {median_width:.0f}, мин. ширина: {min_width:.0f}")
            for i, b in enumerate(text_blocks):
                status = "✓" if b['width'] >= min_width else "✗"
                print(f"  Блок {i}: {b['width']}x{b['height']} {status}")
            print(f"Основных блоков: {len(matching_blocks)} из {len(text_blocks)}")

        return matching_blocks, None

    def _count_max_words_per_row(self, image, bbox, conf_threshold=40):
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return 0, 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        try:
            data = pytesseract.image_to_data(
                gray, lang=self.lang, output_type=Output.DICT, config='--psm 6'
            )
        except Exception:
            return 0, 0

        rows = {}
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            try:
                conf = int(float(data['conf'][i]))
            except (ValueError, TypeError):
                continue
            if conf < conf_threshold:
                continue
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            rows.setdefault(key, []).append(text)

        if not rows:
            return 0, 0

        max_words = 0
        max_word_len = 0
        for tokens in rows.values():
            max_words = max(max_words, len(tokens))
            for tok in tokens:
                letters = re.findall(r'[А-Яа-яЁёA-Za-z]', tok)
                max_word_len = max(max_word_len, len(letters))
        return max_words, max_word_len

    def filter_noise_blocks(self, image, blocks,
                            min_words_per_row=2,
                            min_width_ratio=0.15,
                            min_word_len=4,
                            debug=False):
        if not blocks:
            return blocks

        max_width = max(b['width'] for b in blocks)
        width_threshold = max_width * min_width_ratio

        if debug:
            print(f"[filter_noise] max_width={max_width}, "
                  f"width_threshold={width_threshold:.0f}, "
                  f"min_words_per_row={min_words_per_row}, "
                  f"min_word_len={min_word_len}")

        result = []
        for b in blocks:
            n_words, max_word_len = self._count_max_words_per_row(image, b['bbox'])
            wide_enough = b['width'] >= width_threshold
            long_word_present = max_word_len >= min_word_len
            many_words = n_words >= min_words_per_row and long_word_present

            if wide_enough or many_words:
                result.append(b)
            elif debug:
                print(f"  Блок {b['bbox']} отброшен: "
                      f"слов/ряд={n_words}, longest={max_word_len}, "
                      f"ширина={b['width']} (порог {width_threshold:.0f})")

        if debug:
            print(f"[filter_noise] оставлено {len(result)} из {len(blocks)}")

        return result

    def extract_text_from_block(self, image, block_bbox, debug_words=False,
                                apply_drawing_filter=False):
        x, y, w, h = block_bbox
        block_image = image[y:y+h, x:x+w].copy()

        debug_path = None
        if debug_words and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            debug_path = os.path.join(self.save_dir, f"words_{x}_{y}.png")

        cleaned_image, fraction_texts, fraction_boxes = self.analyze_block_for_fractions(
            block_image,
            frac_height_ratio=self.frac_height_ratio,
            debug_words=debug_words,
            debug_image_path=debug_path
        )

        if apply_drawing_filter:
            cleaned_image = self.filter_drawing_from_roi(cleaned_image)

        config = r'--psm 6 -c preserve_interword_spaces=1'
        if self.char_whitelist:
            config += f' -c tessedit_char_whitelist="{self.char_whitelist}"'
        main_text = pytesseract.image_to_string(cleaned_image, lang=self.lang, config=config).strip()

        final_text = main_text
        if fraction_texts:
            for frac in fraction_texts:
                final_text += f" {frac}"

        has_fraction = bool(fraction_texts)
        return final_text, has_fraction, fraction_boxes

    def clean_text(self, text):
        if not text:
            return ""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_lines = []
        current_paragraph = []
        for line in lines:
            if line.endswith(('.', ':', ';')) or len(line) < 50:
                if current_paragraph:
                    current_paragraph.append(line)
                    cleaned_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                else:
                    cleaned_lines.append(line)
            else:
                current_paragraph.append(line)
        if current_paragraph:
            cleaned_lines.append(' '.join(current_paragraph))
        return '\n'.join(cleaned_lines)

    def _filter_garbage_lines(self, text):
        if not text:
            return ""

        result = []
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            if len(stripped) <= 2:
                continue

            if len(stripped) < 20 and not re.search(r'[А-Яа-яЁёA-Za-z]{2,}', stripped):
                continue

            if re.fullmatch(r'[А-ЯA-Z]?\s*\(\d+:\d+\)', stripped):
                continue

            rare = len(re.findall(r'[|\[\]${}\^\\<>`]', stripped))
            if rare >= 2 and len(stripped) < 40:
                continue

            alnum = len(re.findall(r'[А-Яа-яЁёA-Za-z0-9]', stripped))
            if len(stripped) < 40 and alnum / len(stripped) < 0.6:
                continue

            if self._is_stamp_block(stripped):
                continue

            if re.search(r'(.)\1\1', stripped):
                if not re.search(r'(\.{3,}|-{3,}|_{3,})', stripped):
                    continue

            garbage_words = 0
            for w in re.findall(r'[А-Яа-яЁёA-Za-z]{4,}', stripped):
                if not re.search(r'[аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY]', w):
                    garbage_words += 1
            if garbage_words >= 2:
                continue

            if (len(stripped) < 15
                    and not re.search(r'[.:;]$', stripped)
                    and not re.match(r'^[\d\-*]', stripped)):
                continue

            result.append(line)
        return '\n'.join(result)

    def extract_with_blocks(self, pdf_bytes, file_name):
        images = self.pdf_to_images(pdf_bytes)
        if not images:
            return "", []

        full_text = []
        all_blocks = []

        for page_idx, image in enumerate(images):
            if self.debug:
                print(f"\n--- Страница {page_idx+1} ---")
            text_blocks = self.detect_text_blocks(image)
            if not text_blocks:
                continue

            main_blocks = self.filter_noise_blocks(
                image, text_blocks,
                min_words_per_row=2,
                min_width_ratio=0.15,
                min_word_len=4,
                debug=self.debug,
            )
            if not main_blocks:
                continue

            sorted_blocks = self.sort_blocks_by_columns(main_blocks, 50)

            if self.debug:
                for b in sorted_blocks:
                    print(f"  [sort] bbox={b['bbox']}")

            page_text = []
            for block_info in sorted_blocks:
                block_text, _, _ = self.extract_text_from_block(image, block_info['bbox'])
                cleaned = self.clean_text(block_text)
                if not cleaned:
                    continue

                cleaned = self._filter_garbage_lines(cleaned)
                if not cleaned:
                    if self.debug:
                        print(f"  Блок {block_info['bbox']} полностью отфильтрован как мусор")
                    continue

                if self._is_stamp_block(cleaned):
                    if self.debug:
                        print(f"  Блок {block_info['bbox']} отброшен как штамп/служебное")
                    continue

                page_text.append(cleaned)
                all_blocks.append(cleaned)

            if page_text:
                full_text.append('\n'.join(page_text))

            if not self.production_mode or self.save_images:
                result_image = image.copy()
                for i, block in enumerate(sorted_blocks):
                    _, has_fraction, fraction_boxes = self.extract_text_from_block(
                        image, block['bbox'], debug_words=self.debug
                    )
                    x_blk, y_blk, w_blk, h_blk = block['bbox']
                    blk_color = (255, 0, 255) if has_fraction else (255, 0, 0)
                    cv2.rectangle(result_image, (x_blk, y_blk),
                                  (x_blk + w_blk, y_blk + h_blk), blk_color, 3)
                    label = f'{i+1}'
                    if has_fraction:
                        label += ' (F)'
                    cv2.putText(result_image, label, (x_blk, y_blk - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, blk_color, 3)
                    for (fx, fy, fw, fh) in fraction_boxes:
                        abs_x = x_blk + fx
                        abs_y = y_blk + fy
                        cv2.rectangle(result_image, (abs_x, abs_y),
                                      (abs_x + fw, abs_y + fh), (255, 0, 255), 2)
                save_path = os.path.join(
                    self.save_dir,
                    f"{file_name}_{page_idx+1}_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}.png"
                )
                cv2.imwrite(save_path, result_image)
                logger.debug(f"Сохранено размеченное изображение: {save_path}")

        return '\n\n'.join(full_text), all_blocks

    def extract_text(self, pdf_bytes, file_name):
        text, _ = self.extract_with_blocks(pdf_bytes, file_name)
        return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Визуализация фильтрации блоков OCR")
    parser.add_argument("pdf_path", nargs="?", default=None, help="Путь к PDF-файлу")
    parser.add_argument("--debug", action="store_true", help="Включить отладочный вывод в консоль")
    args = parser.parse_args()

    default_pdf = "/home/user/Desktops/Desktop1/Примеры чертежей/Автокад/КЛАБ.302231.008СБ_180567923.dwg.pdf"
    pdf_path = args.pdf_path or (default_pdf if os.path.exists(default_pdf) else None)
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = input("Введите путь к PDF-файлу: ").strip()
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"❌ Файл не найден: {pdf_path}")
        sys.exit(1)

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    processor = OCRProcessor(dpi=300, lang="rus", psm=6, debug=args.debug, production_mode=False)
    text = processor.extract_text(pdf_bytes, "debug")
    print(text)