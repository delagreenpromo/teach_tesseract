import base64
import os
import sqlite3
from html import escape
from pathlib import Path as FilePath
from typing import Any

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from pdf_processors import PdfProcessingFactory, TokenStorage

BASE_DIR = FilePath(__file__).resolve().parent


def _reload_env():
    load_dotenv(BASE_DIR / ".env", override=False)


_reload_env()


def get_admin_credentials():
    _reload_env()
    login = os.getenv("ADMIN_LOGIN", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")
    return login, password


def get_token_db_path():
    _reload_env()
    return os.getenv("TOKEN_DB_PATH", str(BASE_DIR / "results" / "unique_tokens.db"))


def get_unique_token_stats(db_path=None):
    target_db = db_path or get_token_db_path()
    stats = {"total_unique_tokens": 0, "tokens": []}
    if not os.path.exists(target_db):
        return stats

    with sqlite3.connect(target_db) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS c FROM unique_tokens").fetchone()
        rows = conn.execute(
            "SELECT token, seen_count, source_file, last_seen_at FROM unique_tokens ORDER BY token"
        ).fetchall()
        stats["total_unique_tokens"] = int(total["c"]) if total else 0
        stats["tokens"] = [dict(row) for row in rows]
    return stats


def build_requirement_preview_html(preview_data_url: str | None = None, blocks: list | None = None, original_width: int = 1000, original_height: int = 1000) -> str:
    blocks = blocks or []
    preview_data_url = preview_data_url or ""

    if not preview_data_url:
        preview_data_url = (
            "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='700' %3E"
            "%3Crect width='100%25' height='100%25' fill='%23f8fafc'/%3E%3Ctext x='50%25' y='50%25' "
            "dominant-baseline='middle' text-anchor='middle' font-family='Arial' font-size='32' fill='%230f172a'%3E"
            "Preview unavailable%3C/text%3E%3C/svg%3E"
        )

    overlay_boxes = []
    for idx, block in enumerate(blocks):
        if isinstance(block, dict):
            bbox = block.get("bbox")
        else:
            bbox = None

        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = bbox

        pad_px = 6  # чем больше число, тем шире рамка
        
        x = max(0, x - pad_px)
        y = max(0, y - pad_px)
        w = w + (pad_px * 2)
        h = h + (pad_px * 2)
        # Переводим пиксели в проценты
        left_pct = (x / original_width) * 100
        top_pct = (y / original_height) * 100
        width_pct = (w / original_width) * 100
        height_pct = (h / original_height) * 100

        # ЗДЕСЬ БЫЛА ОШИБКА: добавлен закрывающий тег </div>
        overlay_boxes.append(
            f'<div class="requirement-box" style="left:{left_pct}%; top:{top_pct}%; width:{width_pct}%; height:{height_pct}%;"></div>'
        )

    overlay_html = "".join(overlay_boxes)
    
    # ЗДЕСЬ БЫЛА ОШИБКА: убрано дублирование <div id="fullscreen-preview">
    return f'''
        <div class="card preview-card">
            <div class="preview-header">
                <h3>Миниатюра чертежа</h3>
                <div>
                    <button type="button" class="preview-button" id="toggle-boxes-btn" onclick="toggleBoxes()" style="margin-right: 10px; background: #475569;">Скрыть рамки</button>
                    <button type="button" class="preview-button" onclick="toggleFullscreenPreview()">Открыть в полный экран</button>
                </div>
            </div>
            
            <!-- ВОТ ЭТОТ БЛОК ТЫ СЛУЧАЙНО УДАЛИЛ (картинка и рамки) -->
            <div class="thumbnail-shell">
                <img class="thumbnail-preview" src="{preview_data_url}" alt="Preview of the drawing">
                {overlay_html}
            </div>
        </div>
        
        <div id="fullscreen-preview" class="fullscreen-preview" aria-hidden="true">
            <div class="fullscreen-panel">
                <div class="preview-header modal-header">
                    <h3>Полный просмотр чертежа</h3>
                    <button type="button" class="preview-button" onclick="toggleFullscreenPreview()">Закрыть</button>
                </div>
                <div class="fullscreen-scroll-area">
                    <div class="fullscreen-shell" id="zoom-shell" onclick="this.classList.toggle('zoomed')">
                        <img class="fullscreen-image" src="{preview_data_url}" alt="Full preview of the drawing">
                        {overlay_html}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // НОВАЯ ФУНКЦИЯ: Скрыть/Показать рамки
            function toggleBoxes() {{
                const isHidden = document.body.classList.toggle('hide-boxes');
                const btn = document.getElementById('toggle-boxes-btn');
                if (btn) {{
                    btn.textContent = isHidden ? 'Показать рамки' : 'Скрыть рамки';
                }}
            }}

            function toggleFullscreenPreview() {{
                const modal = document.getElementById('fullscreen-preview');
                if (!modal) return;
                modal.classList.toggle('visible');
                modal.setAttribute('aria-hidden', String(!modal.classList.contains('visible')));
                
                // При закрытии сбрасываем зум
                if (!modal.classList.contains('visible')) {{
                    document.getElementById('zoom-shell').classList.remove('zoomed');
                }}
            }}
        </script>
    '''


def render_pdf_preview(pdf_bytes: bytes, blocks: list | None = None) -> str:
    if fitz is None:
        return ""
    try:
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            if not doc or len(doc) == 0:
                return ""
            page = doc[0]
            # Увеличиваем качество картинки (2.0 вместо 0.6), чтобы при приближении не было пикселей
            zoom = 2.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # БОЛЬШЕ НЕ РИСУЕМ РАМКИ ЧЕРЕЗ FITZ (это делают CSS-стили)
            
            encoded = base64.b64encode(pix.tobytes("png")).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def _preview_blocks(result: dict | None) -> list:
    if not isinstance(result, dict):
        return []
    blocks = result.get("blocks") or []
    normalized = []
    for block in blocks:
        if isinstance(block, dict):
            normalized.append(block)
        else:
            normalized.append({"text": str(block)})
    return normalized


async def process_uploaded_pdf(file_storage):
    if file_storage is None or getattr(file_storage, "filename", "") == "":
        raise ValueError("Пожалуйста, выберите PDF-файл.")

    filename = str(file_storage.filename)
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Поддерживаются только PDF-файлы.")

    if hasattr(file_storage, "read"):
        try:
            pdf_bytes = await file_storage.read()
        except TypeError:
            pdf_bytes = file_storage.read()
    elif hasattr(file_storage, "file"):
        pdf_bytes = file_storage.file.read()
    else:
        raise ValueError("Невозможно прочитать PDF-файл.")

    if not pdf_bytes:
        raise ValueError("Файл пустой.")

    storage = TokenStorage(db_path=get_token_db_path())
    # Идём тем же путём, что и main.py — через PdfProcessingFactory → OCRPdfProcessor
    # → OCRProcessor. ocr_engine не передаём, значит OCRPdfProcessor соберёт его
    # через _build_default_engine() из config.ini — те же настройки, что и в main.py.
    processor = PdfProcessingFactory.create(filename, token_storage=storage)
    if processor is None:
        file_type = filename.lower().split(".")[-2] if "." in filename.lower() else filename.lower()
        raise ValueError(f"Обработка файлов типа '{file_type}' отключена через ENV.")

    result = processor.process(pdf_bytes, source_name=filename)
    preview_data_url = render_pdf_preview(pdf_bytes, _preview_blocks(result))
    if preview_data_url:
        result["preview_data_url"] = preview_data_url
    return result


app = FastAPI(title="Teach Tesseract Admin")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("ADMIN_SECRET_KEY", "teach-tesseract-admin-secret"))


def is_logged_in(request: Request) -> bool:
    return bool(request.session.get("admin_logged_in"))


def render_login_page(error_message: str | None = None) -> HTMLResponse:
    message_block = "" if not error_message else f'<div class="card error">{error_message}</div>'
    return HTMLResponse(
        f"""
        <html lang="ru">
        <head>
            <meta charset="utf-8">
            <title>Admin Login</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
                .panel {{ width: 420px; background: #111827; border: 1px solid #334155; border-radius: 16px; padding: 24px; box-shadow: 0 20px 40px rgba(0,0,0,0.35); }}
                h1 {{ margin-top: 0; font-size: 26px; }}
                form {{ display: flex; flex-direction: column; gap: 12px; }}
                input {{ padding: 12px 10px; border-radius: 10px; border: 1px solid #475569; background: #0f172a; color: #e2e8f0; }}
                button {{ padding: 12px; border: none; border-radius: 10px; background: #3b82f6; color: white; font-weight: bold; cursor: pointer; }}
                .error {{ color: #fca5a5; margin-bottom: 12px; }}
                .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="panel">
                <h1>Админ-панель</h1>
                {message_block}
                <form method="post" action="/login">
                    <input type="text" name="login" placeholder="Логин" required>
                    <input type="password" name="password" placeholder="Пароль" required>
                    <button type="submit">Войти</button>
                </form>
            </div>
        </body>
        </html>
        """
    )


def render_dashboard_page(stats: dict[str, Any]) -> HTMLResponse:
    rows = []
    for row in stats.get("tokens", []):
        rows.append(
            f"<tr><td>{escape(str(row.get('token', '-')))}</td><td>{row.get('seen_count', 0)}</td><td>{escape(str(row.get('source_file', '-')))}</td><td>{escape(str(row.get('last_seen_at', '-')))}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="4">Пока нет накопленных токенов.</td></tr>')

    return HTMLResponse(
        f"""
        <html lang="ru">
        <head>
            <meta charset="utf-8">
            <title>Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; }}
                .topbar {{ display: flex; justify-content: space-between; align-items: center; background: #0f172a; color: white; padding: 16px 24px; }}
                .nav a {{ color: white; text-decoration: none; margin-right: 16px; }}
                .wrap {{ max-width: 1100px; margin: 32px auto; padding: 0 20px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
                .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(15,23,42,0.04); }}
                .big {{ font-size: 32px; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e2e8f0; }}
                th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
                th {{ background: #f1f5f9; }}
            </style>
        </head>
        <body>
            <div class="topbar">
                <div><strong>Teach Tesseract Admin</strong></div>
                <div class="nav">
                    <a href="/dashboard">Статистика</a>
                    <a href="/test">Тестовый режим</a>
                    <a href="/logout">Выход</a>
                </div>
            </div>
            <div class="wrap">
                <h2>Статистика накопленных уникальных токенов</h2>
                <div class="stats">
                    <div class="card">
                        <div>Всего уникальных токенов</div>
                        <div class="big">{stats.get('total_unique_tokens', 0)}</div>
                    </div>
                </div>
                <div class="card">
                    <h3>Токены</h3>
                    <table>
                        <thead>
                            <tr><th>Токен</th><th>Повторы</th><th>Файл</th><th>Последний раз</th></tr>
                        </thead>
                        <tbody>{''.join(rows)}</tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
    )


def render_test_page(result: dict | None = None, error_message: str | None = None) -> HTMLResponse:
    result_html = ""
    if result is not None:
        text = str(result.get("text") or "Нет текста")
        blocks = result.get("blocks") or []
        preview_html = build_requirement_preview_html(
            preview_data_url=result.get("preview_data_url"),
            blocks=_preview_blocks(result),
            original_width=result.get("original_width", 1000),
            original_height=result.get("original_height", 1000),
        )
        result_html = f"""
            <div class="card">
                <h3>Результат обработки</h3>
                <pre>{escape(text)}</pre>
                <p>Блоки: {len(blocks)}</p>
                {preview_html}
            </div>
        """
    error_block = "" if not error_message else f'<div class="card error">{error_message}</div>'

    return HTMLResponse(
        f"""
        <html lang="ru">
        <head>
            <meta charset="utf-8">
            <title>Test mode</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; }}
                .topbar {{ display: flex; justify-content: space-between; align-items: center; background: #0f172a; color: white; padding: 16px 24px; }}
                .nav a {{ color: white; text-decoration: none; margin-right: 16px; }}
                .wrap {{ max-width: 1100px; margin: 32px auto; padding: 0 20px; }}
                .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
                .error {{ color: #b91c1c; }}
                input[type=file] {{ padding: 10px; margin-top: 12px; }}
                button {{ padding: 10px 20px; background: #0f172a; color: white; border: none; border-radius: 10px; cursor: pointer; margin-top: 12px; }}
                pre {{ white-space: pre-wrap; background: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0; padding: 16px; overflow: auto; }}
                .preview-card {{ position: relative; }}
                .preview-header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
                .thumbnail-shell {{ position: relative; display: inline-block; max-width: 100%; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; background: #f8fafc; }}
                .thumbnail-preview {{ display: block; max-width: 100%; height: auto; background: #fff; min-height: 240px; }}
                /* Тонкие рамки без фона */
                /* Стиль полицейской ленты для блоков */
                .requirement-box {{
                    position: absolute;
                    border: 3px solid transparent;
                    border-radius: 4px;
                    /* Красно-белые диагональные полосы */
                    background: repeating-linear-gradient(
                        -45deg,
                        #ef4444,
                        #ef4444 2px,
                        #ffffff 2px,
                        #ffffff 4px
                    ) border-box;
                    /* Магия CSS: скрываем заливку внутри, оставляя только рамку */
                    -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
                    mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
                    -webkit-mask-composite: xor;
                    mask-composite: exclude;
                    /* Легкая тень, чтобы лента читалась на любом фоне */
                    filter: drop-shadow(0px 0px 3px rgba(0, 0, 0, 0.5));
                    pointer-events: none;
                }}
                
                /* Скрываем все рамки, если на body есть класс hide-boxes */
                body.hide-boxes .requirement-box {{
                    display: none !important;
                }}

                .preview-button {{ padding: 9px 14px; border: none; border-radius: 10px; background: #0f172a; color: white; cursor: pointer; }}
                
                /* Модальное окно */
                .fullscreen-preview {{ position: fixed; inset: 0; display: none; background: rgba(15, 23, 42, 0.8); padding: 24px; z-index: 1000; }}
                .fullscreen-preview.visible {{ display: flex; justify-content: center; align-items: center; }}
                
                /* Панель внутри модалки с ползунками */
                .fullscreen-panel {{ position: relative; max-width: 95vw; width: 95vw; max-height: 95vh; height: 95vh; display: flex; flex-direction: column; overflow: hidden; background: white; border-radius: 16px; padding: 18px; box-shadow: 0 30px 60px rgba(0,0,0,0.35); }}
                
                /* Область прокрутки */
                .fullscreen-scroll-area {{ flex-grow: 1; overflow: auto; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0; }}
                
                /* Контейнер, который будет увеличиваться */
                .fullscreen-shell {{ position: relative; display: inline-block; transform-origin: top left; transition: transform 0.2s; margin: 20px; cursor: zoom-in; }}
                .fullscreen-shell.zoomed {{ transform: scale(2.5); cursor: zoom-out; margin: 50px; }}
                .fullscreen-image {{ display: block; max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="topbar">
                <div><strong>Teach Tesseract Admin</strong></div>
                <div class="nav">
                    <a href="/dashboard">Статистика</a>
                    <a href="/test">Тестовый режим</a>
                    <a href="/logout">Выход</a>
                </div>
            </div>
            <div class="wrap">
                <div class="card">
                    <h2>Тестовый режим обработки PDF</h2>
                    <p>Локальная проверка на тестовых данных. Production-часть остаётся прежней.</p>
                    <form method="post" action="/test" enctype="multipart/form-data">
                        <input type="file" name="file" accept="application/pdf" required>
                        <div><button type="submit">Обработать PDF</button></div>
                    </form>
                </div>
                {error_block}
                {result_html}
            </div>
        </body>
        </html>
        """
    )


@app.get("/")
async def root(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")


@app.api_route("/login", methods=["GET", "POST", "HEAD"], response_class=HTMLResponse)
async def login_view(request: Request, login: str | None = Form(default=None), password: str | None = Form(default=None)):
    if request.method == "GET":
        if is_logged_in(request):
            return RedirectResponse(url="/dashboard")
        return render_login_page()
    if login is None or password is None:
        return render_login_page("Введите логин и пароль.")
    admin_login, admin_password = get_admin_credentials()
    if login == admin_login and password == admin_password:
        request.session["admin_logged_in"] = True
        return RedirectResponse(url="/dashboard", status_code=303)
    return render_login_page("Неверный логин или пароль.")


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login")
    return render_dashboard_page(get_unique_token_stats())


@app.api_route("/test", methods=["GET", "POST", "HEAD"], response_class=HTMLResponse)
async def test_view(request: Request, file: UploadFile | None = File(default=None)):
    if not is_logged_in(request):
        return RedirectResponse(url="/login")
    if request.method == "GET":
        return render_test_page()
    if file is None:
        return render_test_page(error_message="Пожалуйста, выберите PDF-файл.")
    try:
        result = await process_uploaded_pdf(file)
        return render_test_page(result=result)
    except Exception as exc:
        return render_test_page(error_message=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ADMIN_PORT", "5002")))