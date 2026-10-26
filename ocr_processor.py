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
        "сертификат",
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
        # Штампы и основная надпись:
        "т.контр.",
        "н.контр.",
        "разраб.",
        "пров.",
        "утв.",
        "изм.",
        "лист",
        "№ докум.",
        "подп.",
        "дата"
    )

    def __init__(self, dpi=300, lang="rus", psm=6, char_whitelist="", min_area_ratio=0.0005, width_tolerance=0.3,
                 save_images=False, save_dir=None, use_advanced=True,
                 kernel_width=40, kernel_height=60, dilation_iterations=2,
                 debug=False, production_mode=True, frac_height_ratio=2.0, word_merge_kernel_width=15,
                 format_settings=None, auto_format=True):
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
        self.format_settings = format_settings or {}
        self.auto_format = auto_format

        if self.save_images or not self.production_mode:
            os.makedirs(self.save_dir, exist_ok=True)

    def _detect_page_format(self, image):
        """Определяет формат страницы по длинной стороне в мм.

        DPI-нормализация: max(w, h) / dpi * 25.4 = длинная сторона в мм.
        Диапазоны подобраны так, чтобы захватывать обрезки/рамки:
        A4: < 350 мм   (A4 = 297 мм)
        A3: 350–480    (A3 = 420 мм)
        A2: 480–650    (A2 = 594 мм)
        A1: 650–900    (A1 = 841 мм)
        A0: >= 900     (A0 = 1189 мм)
        """
        h, w = image.shape[:2]
        max_dim_mm = max(w, h) / self.dpi * 25.4

        if max_dim_mm < 350:
            return "A4"
        if max_dim_mm < 480:
            return "A3"
        if max_dim_mm < 650:
            return "A2"
        if max_dim_mm < 900:
            return "A1"
        return "A0"

    def _resolve_kernel_params(self, image):
        """Возвращает (kernel_width, kernel_height, dilation_iterations) для страницы.

        Приоритет:
        1. per-format настройка (если auto_format=True и формат определён);
        2. общие self.kernel_* (fallback).
        """
        if not self.auto_format or not self.format_settings:
            return self.kernel_width, self.kernel_height, self.dilation_iterations

        fmt = self._detect_page_format(image)
        settings = self.format_settings.get(fmt) or {}

        kw = settings.get("kernel_width", self.kernel_width)
        kh = settings.get("kernel_height", self.kernel_height)
        di = settings.get("dilation_iterations", self.dilation_iterations)

        if self.debug:
            print(f"[auto_format] формат={fmt}, kernel={kw}x{kh}, iter={di}")

        return kw, kh, di

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

        # Если контуров/линий очень много — это точно чертёж, даже если есть текст
        if contour_ratio > 0.4 and hough_ratio > 0.1:
            if self.debug and block_info:
                print(f"  Блок {block_info} отброшен как чертёж (много линий)")
            return True

        # Если текста нет — тоже отбрасываем
        if not self._is_real_text_block(roi):
            if self.debug and block_info:
                print(f"  Блок {block_info} отброшен (нет текста)")
            return True

        # Если текста много и линий мало — оставляем
        if self.debug and block_info:
            print(f"  Блок {block_info} оставлен как текст")
        return False

    def _is_stamp_block(self, text):
        if not text:
            return False
        low = text.lower()
        return any(m in low for m in self._STAMP_MARKERS)

    def _find_fraction_line(self, word_roi):
        """Ищет горизонтальную линию-разделитель дроби внутри ROI.

        Возвращает Y-координату центра линии или None.

        Отсекает линии рамки чертежа, которые на маленьких A4 проходят
        вплотную к тексту и ложно интерпретируются как дробная черта:

        - рамка обычно занимает > 85% ширины ROI;
        - рамка часто идёт по самому краю ROI (top/bottom);
        - у настоящей дробной черты высота 1–2 px, у контура рамки
            высота может разъезжаться при Otsu/морфологии.
        """
        gray = cv2.cvtColor(word_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        roi_w = word_roi.shape[1]
        roi_h = word_roi.shape[0]
        if roi_w < 10 or roi_h < 10:
            return None

        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (roi_w // 3, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best_cnt)

        # 1) Слишком короткая — не дробь
        if w < roi_w * 0.3:
            return None

        # 2) Слишком длинная (почти на всю ширину) — это рамка, а не дробная черта
        if w > roi_w * 0.85:
            return None

        # 3) Контур «толстый» по вертикали — это не линия, а закрашенная область
        if h > max(8, int(roi_h * 0.10)):
            return None

        line_center_y = y + h // 2

        # 4) Линия на самом краю ROI — это рамка/подчёркивание, не дробь
        if line_center_y < roi_h * 0.15 or line_center_y > roi_h * 0.85:
            return None

        return line_center_y

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
                # Защита: не стираем объекты, которые находятся прямо внутри текста
                roi_fragment = roi[y:y+h, x:x+w]
                if self._is_text_fragment(roi_fragment):
                    continue
                
                # Если это очень длинная линия (рамка) — стираем без вопросов
                if max(w, h) > 300:
                    drawing_mask[labels == i] = 255
                    continue

        cleaned = roi.copy()
        cleaned[drawing_mask > 0] = (255, 255, 255)
        return cleaned

    @staticmethod
    def _looks_like_real_fraction(num_text, den_text):
        """True, если num/den — действительно дробь ('1', '2', '5/16', 'Ø8'),
        а не куски длинного обозначения стали/ГОСТа.
        """
        num = (num_text or "").strip()
        den = (den_text or "").strip()
        if not num or not den:
            return False

        # Дробь — это одно короткое «слово» сверху и одно снизу, без длинных фраз
        if len(num.split()) > 1 or len(den.split()) > 1:
            return False

        # Ни числитель, ни знаменатель не должны быть длинными
        if len(num) > 12 or len(den) > 12:
            return False

        # В дроби обычно только цифры/латиница/°, не должно быть «ГОСТ», «МПа», «мм»
        if re.search(r'[А-Яа-яЁё]{3,}', num) or re.search(r'[А-Яа-яЁё]{3,}', den):
            return False

        return True

    def _is_text_fragment(self, fragment):
        if fragment.size == 0:
            return False
        try:
            config = '--psm 7 -c tessedit_char_whitelist=0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя.,:;()"'
            res = pytesseract.image_to_string(fragment, lang=self.lang, config=config).strip()
            return len(res) > 0
        except Exception:
            return False

    def _extract_as_fraction(self, block_image):
        """Ищет горизонтальную линию, режет блок на 2 части, распознаёт их 
        и склеивает в вид 'числитель/знаменатель'.
        
        Возвращает (text, is_fraction).
        """
        gray = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Ищем длинную горизонтальную линию
        h, w = block_image.shape[:2]

        if h > 200 or w > 400:
            return None, False

        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=30,
                                minLineLength=w * 0.3, maxLineGap=5)
        
        if lines is None:
            return None, False

        # Находим саму длинную горизонтальную линию
        best_y = None
        best_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > 3:  # не горизонтальная
                continue
            length = abs(x2 - x1)
            if length > best_len:
                best_len = length
                best_y = (y1 + y2) // 2

        if best_y is None or best_y < h * 0.15 or best_y > h * 0.85:
            return None, False

        # Режем
        top_part = block_image[0:best_y - 2, :]
        bot_part = block_image[best_y + 2:, :]

        if top_part.size == 0 or bot_part.size == 0:
            return None, False

        # Распознаём по отдельности
        config = '--psm 7'
        top_text = pytesseract.image_to_string(top_part, lang=self.lang, config=config).strip()
        bot_text = pytesseract.image_to_string(bot_part, lang=self.lang, config=config).strip()

        # Чистим от мусора
        top_clean = re.sub(r'[^\w.,\-]', '', top_text)
        bot_clean = re.sub(r'[^\w.,\-]', '', bot_text)

        # Если обе части содержат текст — это дробь
        if top_clean and bot_clean:
            return f"{top_clean}/{bot_clean}", True

        return None, False

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
                    # линии-разделителя нет — это не дробь (например, высокий логотип)
                    continue

                top_part = word_roi[0:line_y, :]
                bot_part = word_roi[line_y:, :]

                num_text = pytesseract.image_to_string(top_part, lang=self.lang, config='--psm 7').strip()
                den_text = pytesseract.image_to_string(bot_part, lang=self.lang, config='--psm 7').strip()
                num_clean = re.sub(r'[^\w.,\-]', '', num_text)
                den_clean = re.sub(r'[^\w.,\-]', '', den_text)

                if not num_clean or not den_clean:
                    continue

                # <<< ГЛАВНЫЙ ФИКС >>>
                # Если это не дробь, а длинный текст (обозначение стали), оставляем
                # high_word на месте — пусть image_to_string прочитает его как обычный
                # текст блока в правильном порядке. НЕ вырезаем, НЕ приклеиваем как "num/den".
                if not self._looks_like_real_fraction(num_clean, den_clean):
                    continue

                fraction_texts.append(f"{num_clean}/{den_clean}")
                cleaned_img[y1:y2, x1:x2] = (255, 255, 255)
                high_word_boxes.append((wd['x'], wd['y'], wd['w'], wd['h']))

            return cleaned_img, fraction_texts, high_word_boxes

        return block_image, [], []

    def _extract_ordered_text(self, block_image):
        if block_image.size == 0:
            return ""

        def get_text(cfg):
            try:
                data = pytesseract.image_to_data(
                    block_image, lang=self.lang, output_type=Output.DICT, config=cfg
                )
            except Exception:
                return ""
            
            lines = {}
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
                key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                entry = lines.setdefault(
                    key, {'words': [], 'top': float('inf'), 'left': float('inf')}
                )
                entry['words'].append(text)
                entry['top'] = min(entry['top'], data['top'][i])
                entry['left'] = min(entry['left'], data['left'][i])

            ordered = sorted(lines.values(), key=lambda l: (l['top'], l['left']))
            return '\n'.join(' '.join(l['words']) for l in ordered)

        # 1. Пробуем стандартный PSM 6
        cfg6 = r'--psm 6 -c preserve_interword_spaces=1'
        if self.char_whitelist:
            cfg6 += f' -c tessedit_char_whitelist="{self.char_whitelist}"'
        text = get_text(cfg6)

        # 2. Если текста слишком мало (значит, PSM 6 не справился с широким блоком) — включаем PSM 11
        if len(text.strip()) < 15:
            cfg11 = r'--psm 11 -c preserve_interword_spaces=1'
            if self.char_whitelist:
                cfg11 += f' -c tessedit_char_whitelist="{self.char_whitelist}"'
            text_11 = get_text(cfg11)
            if len(text_11.strip()) > len(text.strip()):
                text = text_11

        return text

    def _is_stamp_roi(self, roi):
        """Быстрая проверка: если в блоке есть текст подписи — выкидываем его."""
        try:
            # Берем только 1 строку для скорости
            text = pytesseract.image_to_string(roi, lang=self.lang, config='--psm 6').lower()
            return self._is_stamp_block(text)
        except Exception:
            return False

    def detect_text_blocks(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        binary = self.preprocess_image(img_rgb)

        data = pytesseract.image_to_data(img_rgb, lang=self.lang, output_type=Output.DICT,
                                         config='--psm 11')

        mask = np.zeros_like(binary)
        for i in range(len(data["level"])):
            if data["text"][i].strip():
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                mask[y:y+h, x:x+w] = 255

        kw, kh, di = self._resolve_kernel_params(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        dilated = cv2.dilate(mask, kernel, iterations=di)

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

            if self._is_stamp_roi(roi):
                if self.debug:
                    print(f"  Блок #{idx} отброшен как ЭЦП/штамп")
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

    def _merge_adjacent_boxes(self, blocks):
        if not blocks:
            return []

        sorted_blocks = sorted(blocks, key=lambda b: b['bbox'][1])
        merged = []

        for block in sorted_blocks:
            x, y, w, h = block['bbox']
            merged_into_existing = False

            for i, existing in enumerate(merged):
                ex_x, ex_y, ex_w, ex_h = existing['bbox']

                # Вычисляем расстояние между краями
                gap_x = max(0, max(x, ex_x) - min(x + w, ex_x + ex_w))
                gap_y = max(0, max(y, ex_y) - min(y + h, ex_y + ex_h))

                # Если блоки находятся рядом (ближе 150 px по Y и ближе 300 px по X)
                if gap_y < 200 and gap_x < 1000:
                    
                    # Не склеиваем, если один из блоков — это огромный чертёж
                    if max(w, h) > 2000 or max(ex_w, ex_h) > 2000:
                        continue

                    new_x = min(x, ex_x)
                    new_y = min(y, ex_y)
                    new_w = max(x + w, ex_x + ex_w) - new_x
                    new_h = max(y + h, ex_y + ex_h) - new_y

                    merged[i]['bbox'] = (new_x, new_y, new_w, new_h)
                    merged[i]['width'] = new_w
                    merged[i]['height'] = new_h
                    merged_into_existing = True
                    break

            if not merged_into_existing:
                merged.append(block)

        return merged


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
        
        # Делаем одинаковые небольшие отступы, чтобы не захватывать линии рамки
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        block_image = image[y1:y2, x1:x2].copy()

        debug_path = None
        if debug_words and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            debug_path = os.path.join(self.save_dir, f"words_{x}_{y}.png")

        # <<< НОВОЕ: пробуем распознать как дробь >>>
        fraction_text, is_fraction = self._extract_as_fraction(block_image)
        if is_fraction:
            return fraction_text, True, []

        # Если не дробь — обычный пайплайн
        cleaned_image, fraction_texts, fraction_boxes = self.analyze_block_for_fractions(
            block_image,
            frac_height_ratio=self.frac_height_ratio,
            debug_words=debug_words,
            debug_image_path=debug_path
        )

        if apply_drawing_filter:
            cleaned_image = self.filter_drawing_from_roi(cleaned_image)

        main_text = self._extract_ordered_text(cleaned_image)

        final_text = main_text
        if fraction_texts:
            for frac in fraction_texts:
                final_text += f" {frac}"

        return final_text, bool(fraction_texts), fraction_boxes

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

    def _suppress_overlapping_boxes(self, blocks):
        """Удаляет блоки, которые сильно накладываются друг на друга (IoU > 0.3)."""
        if not blocks:
            return []
        
        # Сортируем по площади (от больших к меньшим)
        sorted_blocks = sorted(blocks, key=lambda b: b['width'] * b['height'], reverse=True)
        result = []
        
        for block in sorted_blocks:
            x1, y1, w1, h1 = block['bbox']
            x2, y2 = x1 + w1, y1 + h1
            
            keep = True
            for existing in result:
                ex1, ey1, ew1, eh1 = existing['bbox']
                ex2, ey2 = ex1 + ew1, ey1 + eh1
                
                inter_x1 = max(x1, ex1)
                inter_y1 = max(y1, ey1)
                inter_x2 = min(x2, ex2)
                inter_y2 = min(y2, ey2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = w1 * h1
                    area2 = ew1 * eh1
                    
                    # Если блок на 30% лежит внутри другого, считаем его дублем
                    if inter_area / min(area1, area2) > 0.3:
                        keep = False
                        break
            
            if keep:
                result.append(block)
                
        return result

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

            text_blocks = self._merge_adjacent_boxes(text_blocks)
            text_blocks = self._suppress_overlapping_boxes(text_blocks)

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
                block_text, _, _ = self.extract_text_from_block(
                    image, 
                    block_info['bbox'], 
                    apply_drawing_filter=True)
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

    default_pdf = ""
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