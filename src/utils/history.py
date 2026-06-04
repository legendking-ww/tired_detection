"""检测历史数据持久化：记录每次统计窗口的疲劳指标与告警事件。

表结构：
- detection_history: 每个统计窗口一条记录，包含 EAR/MAR/姿态/PERCLOS/眨眼率/融合分等
- alert_events: 强告警事件单独记录（含触发上下文）

使用方式：
    from src.utils.history import DetectionHistory

    hist = DetectionHistory("mrsoft.db")
    hist.start_session()
    # 每个统计窗口：
    hist.record_window(timestamp, ear, mar, pitch, perclos, blink_rate,
                       visual_score, audio_score, fused_score, alert_level)
    # 告警时：
    hist.record_alert(timestamp, alert_type, fused_score, context)
    hist.end_session()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from src.utils.logger import get_logger

_log = get_logger(__name__)

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS detection_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    ear         REAL,
    mar         REAL,
    pitch       REAL,
    perclos     REAL,
    blink_rate  REAL,
    visual_score REAL,
    audio_score REAL,
    fused_score REAL,
    alert_level TEXT,
    extra       TEXT
)
"""

_CREATE_ALERT_TABLE = """
CREATE TABLE IF NOT EXISTS alert_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    alert_type  TEXT    NOT NULL,
    fused_score REAL,
    context     TEXT
)
"""

_CREATE_INDEX_HISTORY_SESSION = """
CREATE INDEX IF NOT EXISTS idx_hist_session ON detection_history(session_id, timestamp)
"""

_CREATE_INDEX_ALERT_SESSION = """
CREATE INDEX IF NOT EXISTS idx_alert_session ON alert_events(session_id, timestamp)
"""


class DetectionHistory:
    """线程安全的检测历史记录器。"""

    def __init__(self, db_path: str = "mrsoft.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._session_id: str | None = None

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_tables(self) -> None:
        """创建历史记录表（首次使用时调用）。"""
        try:
            conn = self._get_conn()
            conn.execute(_CREATE_HISTORY_TABLE)
            conn.execute(_CREATE_ALERT_TABLE)
            conn.execute(_CREATE_INDEX_HISTORY_SESSION)
            conn.execute(_CREATE_INDEX_ALERT_SESSION)
            conn.commit()
            conn.close()
            _log.info("历史数据表初始化完成")
        except Exception as e:
            _log.error("初始化历史数据表失败: %s", e)

    def start_session(self) -> str:
        """开始新检测会话，返回 session_id。"""
        with self._lock:
            self._session_id = f"{int(time.time() * 1000)}_{id(self) % 100000:05d}"
            _log.info("历史会话开始: %s", self._session_id)
            return self._session_id

    def end_session(self) -> None:
        """结束当前会话。"""
        with self._lock:
            if self._session_id:
                _log.info("历史会话结束: %s", self._session_id)
            self._session_id = None

    def record_window(
        self,
        ts: float,
        ear: float | None,
        mar: float | None,
        pitch: float | None,
        perclos: float | None,
        blink_rate: float | None,
        visual_score: float | None,
        audio_score: float | None,
        fused_score: float | None,
        alert_level: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """记录一个统计窗口的快照。"""
        sid = self._session_id
        if not sid:
            return
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO detection_history
                   (session_id, timestamp, ear, mar, pitch, perclos, blink_rate,
                    visual_score, audio_score, fused_score, alert_level, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sid,
                    ts,
                    ear,
                    mar,
                    pitch,
                    perclos,
                    blink_rate,
                    visual_score,
                    audio_score,
                    fused_score,
                    alert_level,
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            _log.warning("记录窗口数据失败: %s", e)

    def record_alert(
        self,
        ts: float,
        alert_type: str,
        fused_score: float | None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """记录一次强告警事件。"""
        sid = self._session_id
        if not sid:
            return
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO alert_events (session_id, timestamp, alert_type, fused_score, context)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    sid,
                    ts,
                    alert_type,
                    fused_score,
                    json.dumps(context, ensure_ascii=False) if context else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            _log.warning("记录告警事件失败: %s", e)

    # ---- 查询接口 ----

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """列出最近 N 个会话摘要。"""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT session_id,
                          MIN(timestamp) AS start_ts,
                          MAX(timestamp) AS end_ts,
                          COUNT(*)      AS window_count,
                          AVG(fused_score) AS avg_fused,
                          MAX(fused_score) AS max_fused,
                          SUM(CASE WHEN alert_level='danger' THEN 1 ELSE 0 END) AS danger_count
                   FROM detection_history
                   GROUP BY session_id
                   ORDER BY start_ts DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            conn.close()
            return [
                {
                    "session_id": r[0],
                    "start_ts": r[1],
                    "end_ts": r[2],
                    "window_count": r[3],
                    "avg_fused": round(r[4], 3) if r[4] is not None else 0,
                    "max_fused": round(r[5], 3) if r[5] is not None else 0,
                    "danger_count": r[6] or 0,
                }
                for r in rows
            ]
        except Exception as e:
            _log.warning("查询会话列表失败: %s", e)
            return []

    def get_session_data(self, session_id: str) -> list[dict[str, Any]]:
        """返回指定会话的所有窗口数据。"""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT timestamp, ear, mar, pitch, perclos, blink_rate,
                          visual_score, audio_score, fused_score, alert_level
                   FROM detection_history
                   WHERE session_id = ?
                   ORDER BY timestamp""",
                (session_id,),
            ).fetchall()
            conn.close()
            return [
                {
                    "timestamp": r[0],
                    "ear": r[1],
                    "mar": r[2],
                    "pitch": r[3],
                    "perclos": r[4],
                    "blink_rate": r[5],
                    "visual_score": r[6],
                    "audio_score": r[7],
                    "fused_score": r[8],
                    "alert_level": r[9],
                }
                for r in rows
            ]
        except Exception as e:
            _log.warning("查询会话数据失败: %s", e)
            return []

    def get_alert_events(self, session_id: str) -> list[dict[str, Any]]:
        """返回指定会话的所有告警事件。"""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT timestamp, alert_type, fused_score, context
                   FROM alert_events
                   WHERE session_id = ?
                   ORDER BY timestamp""",
                (session_id,),
            ).fetchall()
            conn.close()
            return [
                {
                    "timestamp": r[0],
                    "alert_type": r[1],
                    "fused_score": r[2],
                    "context": r[3],
                }
                for r in rows
            ]
        except Exception as e:
            _log.warning("查询告警事件失败: %s", e)
            return []
