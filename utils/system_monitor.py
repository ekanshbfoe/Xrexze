"""Xrexze System Monitor — psutil wrappers for telemetry."""

import psutil


def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=0)


def get_ram_usage() -> tuple[float, float, float]:
    mem = psutil.virtual_memory()
    return (
        mem.used / (1024 ** 3),
        mem.total / (1024 ** 3),
        mem.percent,
    )


def is_ram_safe(threshold_gb: float = 6.0) -> bool:
    used_gb, _, _ = get_ram_usage()
    return used_gb < threshold_gb


def get_system_summary() -> dict:
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    return {
        "cpu_cores": cpu_count,
        "cpu_freq_mhz": cpu_freq.current if cpu_freq else 0,
        "ram_total_gb": round(mem.total / (1024 ** 3), 1),
        "ram_available_gb": round(mem.available / (1024 ** 3), 1),
        "ram_percent": mem.percent,
    }
