import logging
import os
import inspect
import config_env as cfg

# Создаём логгер
logger = logging.getLogger("custom_logger")
logger.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))

# Настраиваем логгирование только один раз
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # Консольный хендлер (stdout → docker logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый хендлер
    file_handler = logging.FileHandler(cfg.LOG_FILE, encoding='utf-8', mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Вспомогательная функция для префикса
def _get_log_prefix():
    frame = inspect.stack()[2]
    cls = frame.frame.f_locals.get('self', None)
    if cls:
        cls_name = cls.__class__.__name__
        method_name = frame.function
        return f"{cls_name}.{method_name}"
    else:
        return f"{frame.function} ({frame.filename}:{frame.lineno})"

# Уровни логирования
def print_debug(message: str):
    logger.debug(f"{_get_log_prefix()} - {message}")

def print_info(message: str):
    logger.info(f"{_get_log_prefix()} - {message}")

def print_warning(message: str):
    logger.warning(f"{_get_log_prefix()} - {message}")

def print_error(message: str):
    logger.error(f"{_get_log_prefix()} - {message}")

def print_critical(message: str):
    logger.critical(f"{_get_log_prefix()} - {message}")
