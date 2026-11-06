"""
Structured Logging with JSON formatting, execution timing decorators, and privacy data masking.
"""
import re
import time
import json
import logging
from pathlib import Path
from functools import wraps
from typing import Dict, Any, Callable, Optional


class DataMasker:
    """Masks personal identifiers (emails, phone numbers) before writing logs."""
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_REGEX = re.compile(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b")

    @classmethod
    def mask(cls, text: str) -> str:
        if not isinstance(text, str):
            return text
        masked = cls.EMAIL_REGEX.sub("[EMAIL_MASKED]", text)
        masked = cls.PHONE_REGEX.sub("[PHONE_MASKED]", masked)
        return masked


class StructuredLogger:
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(__file__).resolve().parent.parent / "experiments"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "structured_logs.jsonl"

    def log(
        self,
        level: str,
        module_name: str,
        message: str,
        agent_id: str = "core_pipeline",
        execution_time_ms: float = 0.0,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Writes a structured JSON log entry."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level.upper(),
            "module_name": module_name,
            "agent_id": agent_id,
            "message": DataMasker.mask(message),
            "execution_time_ms": round(execution_time_ms, 2),
            "extra": extra or {}
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def info(self, module: str, message: str, agent_id: str = "agent", duration_ms: float = 0.0):
        self.log("INFO", module, message, agent_id, duration_ms)

    def warning(self, module: str, message: str, agent_id: str = "agent"):
        self.log("WARNING", module, message, agent_id)

    def error(self, module: str, message: str, agent_id: str = "agent", extra: Optional[Dict[str, Any]] = None):
        self.log("ERROR", module, message, agent_id, extra=extra)


# Global logger instance
logger = StructuredLogger()


def timer(module_name: str = "function", agent_id: str = "timer_wrapper"):
    """Decorator to measure and log execution duration in milliseconds."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                logger.info(module_name, f"Completed '{func.__name__}'", agent_id=agent_id, duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                logger.error(module_name, f"Failed '{func.__name__}': {str(e)}", agent_id=agent_id, extra={"error": str(e)})
                raise
        return wrapper
    return decorator
