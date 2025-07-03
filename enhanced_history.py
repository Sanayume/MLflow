"""
Enhanced History Management System
增强的历史记录管理系统
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HistoryType(Enum):
    """历史记录类型"""
    CONVERSATION = "conversation"
    CODE_EXECUTION = "code_execution"
    ML_RESULT = "ml_result"
    FILE_OPERATION = "file_operation"
    SYSTEM_EVENT = "system_event"
    USER_ACTION = "user_action"
    ERROR_EVENT = "error_event"

class EventSeverity(Enum):
    """事件严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    DEBUG = "debug"

@dataclass
class HistoryEntry:
    """历史记录条目"""
    entry_id: str
    timestamp: datetime.datetime
    history_type: HistoryType
    severity: EventSeverity
    title: str
    description: str
    metadata: Dict[str, Any]
    tags: List[str]
    session_id: str
    user_id: Optional[str] = None
    execution_id: Optional[str] = None
    file_paths: List[str] = None
    duration_ms: Optional[int] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['history_type'] = self.history_type.value
        result['severity'] = self.severity.value
        result['file_paths'] = self.file_paths or []
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """从字典创建"""
        return cls(
            entry_id=data['entry_id'],
            timestamp=datetime.datetime.fromisoformat(data['timestamp']),
            history_type=HistoryType(data['history_type']),
            severity=EventSeverity(data['severity']),
            title=data['title'],
            description=data['description'],
            metadata=data['metadata'],
            tags=data['tags'],
            session_id=data['session_id'],
            user_id=data.get('user_id'),
            execution_id=data.get('execution_id'),
            file_paths=data.get('file_paths', []),
            duration_ms=data.get('duration_ms'),
            success=data.get('success', True)
        )

@dataclass
class SearchFilters:
    """搜索过滤器"""
    history_types: List[HistoryType] = None
    severities: List[EventSeverity] = None
    tags: List[str] = None
    session_ids: List[str] = None
    user_ids: List[str] = None
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    search_text: str = None
    success_only: bool = None
    execution_ids: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'history_types': [ht.value for ht in self.history_types] if self.history_types else None,
            'severities': [s.value for s in self.severities] if self.severities else None,
            'tags': self.tags,
            'session_ids': self.session_ids,
            'user_ids': self.user_ids,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'search_text': self.search_text,
            'success_only': self.success_only,
            'execution_ids': self.execution_ids
        }

class EnhancedHistoryManager:
    """增强的历史记录管理器"""
    
    def __init__(self, db_path: str = "agent_workspace/history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # 内存缓存
        self._memory_cache: Dict[str, HistoryEntry] = {}
        self._cache_size_limit = 1000
        
        logger.info(f"Enhanced History Manager initialized with database: {self.db_path}")
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建主历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    history_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    execution_id TEXT,
                    file_paths TEXT,
                    duration_ms INTEGER,
                    success BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON history_entries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON history_entries(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_type ON history_entries(history_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_execution_id ON history_entries(execution_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON history_entries(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_success ON history_entries(success)')
            
            # 创建会话统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_stats (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_entries INTEGER DEFAULT 0,
                    conversation_count INTEGER DEFAULT 0,
                    execution_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    user_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建标签统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tag_stats (
                    tag TEXT PRIMARY KEY,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_entry(self, entry: HistoryEntry) -> str:
        """添加历史记录条目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 插入主记录
                cursor.execute('''
                    INSERT INTO history_entries (
                        entry_id, timestamp, history_type, severity, title, 
                        description, metadata, tags, session_id, user_id,
                        execution_id, file_paths, duration_ms, success
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.history_type.value,
                    entry.severity.value,
                    entry.title,
                    entry.description,
                    json.dumps(entry.metadata, ensure_ascii=False),
                    json.dumps(entry.tags, ensure_ascii=False),
                    entry.session_id,
                    entry.user_id,
                    entry.execution_id,
                    json.dumps(entry.file_paths or [], ensure_ascii=False),
                    entry.duration_ms,
                    entry.success
                ))
                
                # 更新会话统计
                self._update_session_stats(cursor, entry)
                
                # 更新标签统计
                self._update_tag_stats(cursor, entry.tags)
                
                conn.commit()
            
            # 添加到内存缓存
            self._add_to_cache(entry)
            
            logger.debug(f"Added history entry: {entry.entry_id}")
            return entry.entry_id
            
        except Exception as e:
            logger.error(f"Error adding history entry: {str(e)}", exc_info=True)
            raise
    
    def _update_session_stats(self, cursor, entry: HistoryEntry):
        """更新会话统计"""
        # 检查会话是否存在
        cursor.execute('SELECT session_id FROM session_stats WHERE session_id = ?', (entry.session_id,))
        exists = cursor.fetchone()
        
        if exists:
            # 更新现有会话
            cursor.execute('''
                UPDATE session_stats SET
                    end_time = ?,
                    total_entries = total_entries + 1,
                    conversation_count = conversation_count + ?,
                    execution_count = execution_count + ?,
                    error_count = error_count + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (
                entry.timestamp.isoformat(),
                1 if entry.history_type == HistoryType.CONVERSATION else 0,
                1 if entry.history_type == HistoryType.CODE_EXECUTION else 0,
                1 if entry.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL] else 0,
                entry.session_id
            ))
        else:
            # 创建新会话
            cursor.execute('''
                INSERT INTO session_stats (
                    session_id, start_time, end_time, total_entries,
                    conversation_count, execution_count, error_count, user_id
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            ''', (
                entry.session_id,
                entry.timestamp.isoformat(),
                entry.timestamp.isoformat(),
                1 if entry.history_type == HistoryType.CONVERSATION else 0,
                1 if entry.history_type == HistoryType.CODE_EXECUTION else 0,
                1 if entry.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL] else 0,
                entry.user_id
            ))
        
        # 更新成功率
        cursor.execute('''
            UPDATE session_stats SET
                success_rate = (
                    SELECT CAST(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
                    FROM history_entries 
                    WHERE session_id = ?
                )
            WHERE session_id = ?
        ''', (entry.session_id, entry.session_id))
    
    def _update_tag_stats(self, cursor, tags: List[str]):
        """更新标签统计"""
        for tag in tags:
            cursor.execute('''
                INSERT OR REPLACE INTO tag_stats (tag, usage_count, last_used)
                VALUES (?, COALESCE((SELECT usage_count FROM tag_stats WHERE tag = ?), 0) + 1, ?)
            ''', (tag, tag, datetime.datetime.now().isoformat()))
    
    def _add_to_cache(self, entry: HistoryEntry):
        """添加到内存缓存"""
        self._memory_cache[entry.entry_id] = entry
        
        # 限制缓存大小
        if len(self._memory_cache) > self._cache_size_limit:
            # 移除最旧的条目
            oldest_key = min(self._memory_cache.keys(), 
                           key=lambda k: self._memory_cache[k].timestamp)
            del self._memory_cache[oldest_key]
    
    def search_entries(self, filters: SearchFilters = None, 
                      limit: int = 100, offset: int = 0,
                      order_by: str = "timestamp", order_desc: bool = True) -> Tuple[List[HistoryEntry], int]:
        """搜索历史记录条目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建查询
                query_parts = ["SELECT * FROM history_entries"]
                count_query_parts = ["SELECT COUNT(*) FROM history_entries"]
                where_clauses = []
                params = []
                
                if filters:
                    # 历史类型过滤
                    if filters.history_types:
                        placeholders = ','.join(['?' for _ in filters.history_types])
                        where_clauses.append(f"history_type IN ({placeholders})")
                        params.extend([ht.value for ht in filters.history_types])
                    
                    # 严重程度过滤
                    if filters.severities:
                        placeholders = ','.join(['?' for _ in filters.severities])
                        where_clauses.append(f"severity IN ({placeholders})")
                        params.extend([s.value for s in filters.severities])
                    
                    # 会话ID过滤
                    if filters.session_ids:
                        placeholders = ','.join(['?' for _ in filters.session_ids])
                        where_clauses.append(f"session_id IN ({placeholders})")
                        params.extend(filters.session_ids)
                    
                    # 用户ID过滤
                    if filters.user_ids:
                        placeholders = ','.join(['?' for _ in filters.user_ids])
                        where_clauses.append(f"user_id IN ({placeholders})")
                        params.extend(filters.user_ids)
                    
                    # 执行ID过滤
                    if filters.execution_ids:
                        placeholders = ','.join(['?' for _ in filters.execution_ids])
                        where_clauses.append(f"execution_id IN ({placeholders})")
                        params.extend(filters.execution_ids)
                    
                    # 时间范围过滤
                    if filters.start_time:
                        where_clauses.append("timestamp >= ?")
                        params.append(filters.start_time.isoformat())
                    
                    if filters.end_time:
                        where_clauses.append("timestamp <= ?")
                        params.append(filters.end_time.isoformat())
                    
                    # 成功状态过滤
                    if filters.success_only is not None:
                        where_clauses.append("success = ?")
                        params.append(filters.success_only)
                    
                    # 文本搜索
                    if filters.search_text:
                        where_clauses.append("(title LIKE ? OR description LIKE ?)")
                        search_pattern = f"%{filters.search_text}%"
                        params.extend([search_pattern, search_pattern])
                    
                    # 标签过滤
                    if filters.tags:
                        for tag in filters.tags:
                            where_clauses.append("tags LIKE ?")
                            params.append(f'%"{tag}"%')
                
                # 添加WHERE子句
                if where_clauses:
                    where_clause = " WHERE " + " AND ".join(where_clauses)
                    query_parts.append(where_clause)
                    count_query_parts.append(where_clause)
                
                # 获取总数
                count_query = " ".join(count_query_parts)
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]
                
                # 添加排序
                order_direction = "DESC" if order_desc else "ASC"
                query_parts.append(f"ORDER BY {order_by} {order_direction}")
                
                # 添加分页
                query_parts.append("LIMIT ? OFFSET ?")
                params.extend([limit, offset])
                
                # 执行查询
                query = " ".join(query_parts)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # 转换为HistoryEntry对象
                entries = []
                for row in rows:
                    entry_data = {
                        'entry_id': row[0],
                        'timestamp': row[1],
                        'history_type': row[2],
                        'severity': row[3],
                        'title': row[4],
                        'description': row[5],
                        'metadata': json.loads(row[6]),
                        'tags': json.loads(row[7]),
                        'session_id': row[8],
                        'user_id': row[9],
                        'execution_id': row[10],
                        'file_paths': json.loads(row[11]) if row[11] else [],
                        'duration_ms': row[12],
                        'success': bool(row[13])
                    }
                    entries.append(HistoryEntry.from_dict(entry_data))
                
                logger.debug(f"Found {len(entries)} entries (total: {total_count})")
                return entries, total_count
                
        except Exception as e:
            logger.error(f"Error searching history entries: {str(e)}", exc_info=True)
            return [], 0
    
    def get_entry_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """根据ID获取历史记录条目"""
        # 首先检查缓存
        if entry_id in self._memory_cache:
            return self._memory_cache[entry_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM history_entries WHERE entry_id = ?', (entry_id,))
                row = cursor.fetchone()
                
                if row:
                    entry_data = {
                        'entry_id': row[0],
                        'timestamp': row[1],
                        'history_type': row[2],
                        'severity': row[3],
                        'title': row[4],
                        'description': row[5],
                        'metadata': json.loads(row[6]),
                        'tags': json.loads(row[7]),
                        'session_id': row[8],
                        'user_id': row[9],
                        'execution_id': row[10],
                        'file_paths': json.loads(row[11]) if row[11] else [],
                        'duration_ms': row[12],
                        'success': bool(row[13])
                    }
                    entry = HistoryEntry.from_dict(entry_data)
                    self._add_to_cache(entry)
                    return entry
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting entry by ID: {str(e)}", exc_info=True)
            return None
    
    def get_session_stats(self, session_id: str = None) -> Dict[str, Any]:
        """获取会话统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute('SELECT * FROM session_stats WHERE session_id = ?', (session_id,))
                    row = cursor.fetchone()
                    if row:
                        return {
                            'session_id': row[0],
                            'start_time': row[1],
                            'end_time': row[2],
                            'total_entries': row[3],
                            'conversation_count': row[4],
                            'execution_count': row[5],
                            'error_count': row[6],
                            'success_rate': row[7],
                            'user_id': row[8]
                        }
                    return {}
                else:
                    # 返回所有会话的汇总
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_sessions,
                            SUM(total_entries) as total_entries,
                            SUM(conversation_count) as total_conversations,
                            SUM(execution_count) as total_executions,
                            SUM(error_count) as total_errors,
                            AVG(success_rate) as avg_success_rate
                        FROM session_stats
                    ''')
                    row = cursor.fetchone()
                    return {
                        'total_sessions': row[0] or 0,
                        'total_entries': row[1] or 0,
                        'total_conversations': row[2] or 0,
                        'total_executions': row[3] or 0,
                        'total_errors': row[4] or 0,
                        'avg_success_rate': row[5] or 0.0
                    }
                    
        except Exception as e:
            logger.error(f"Error getting session stats: {str(e)}", exc_info=True)
            return {}
    
    def get_top_tags(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取使用最多的标签"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT tag, usage_count, last_used 
                    FROM tag_stats 
                    ORDER BY usage_count DESC 
                    LIMIT ?
                ''', (limit,))
                
                return [
                    {
                        'tag': row[0],
                        'usage_count': row[1],
                        'last_used': row[2]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Error getting top tags: {str(e)}", exc_info=True)
            return []
    
    def get_activity_timeline(self, session_id: str = None, 
                            hours: int = 24) -> List[Dict[str, Any]]:
        """获取活动时间线"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 计算时间范围
                end_time = datetime.datetime.now()
                start_time = end_time - datetime.timedelta(hours=hours)
                
                params = [start_time.isoformat(), end_time.isoformat()]
                where_clause = "timestamp BETWEEN ? AND ?"
                
                if session_id:
                    where_clause += " AND session_id = ?"
                    params.append(session_id)
                
                cursor.execute(f'''
                    SELECT 
                        datetime(timestamp) as hour,
                        history_type,
                        severity,
                        COUNT(*) as count
                    FROM history_entries 
                    WHERE {where_clause}
                    GROUP BY datetime(timestamp), history_type, severity
                    ORDER BY timestamp
                ''', params)
                
                timeline = []
                for row in cursor.fetchall():
                    timeline.append({
                        'hour': row[0],
                        'history_type': row[1],
                        'severity': row[2],
                        'count': row[3]
                    })
                
                return timeline
                
        except Exception as e:
            logger.error(f"Error getting activity timeline: {str(e)}", exc_info=True)
            return []
    
    def export_history(self, filters: SearchFilters = None, 
                      format: str = "json") -> str:
        """导出历史记录"""
        try:
            entries, _ = self.search_entries(filters, limit=10000)
            
            if format == "json":
                export_data = {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'total_entries': len(entries),
                    'filters': filters.to_dict() if filters else None,
                    'entries': [entry.to_dict() for entry in entries]
                }
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            
            elif format == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # 写入标题行
                writer.writerow([
                    'entry_id', 'timestamp', 'history_type', 'severity',
                    'title', 'description', 'session_id', 'user_id',
                    'execution_id', 'duration_ms', 'success', 'tags'
                ])
                
                # 写入数据行
                for entry in entries:
                    writer.writerow([
                        entry.entry_id,
                        entry.timestamp.isoformat(),
                        entry.history_type.value,
                        entry.severity.value,
                        entry.title,
                        entry.description,
                        entry.session_id,
                        entry.user_id,
                        entry.execution_id,
                        entry.duration_ms,
                        entry.success,
                        ','.join(entry.tags)
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting history: {str(e)}", exc_info=True)
            raise
    
    def cleanup_old_entries(self, days: int = 30):
        """清理旧的历史记录"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除旧记录
                cursor.execute(
                    'DELETE FROM history_entries WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                
                # 清理会话统计（如果没有相关的历史记录）
                cursor.execute('''
                    DELETE FROM session_stats 
                    WHERE session_id NOT IN (
                        SELECT DISTINCT session_id FROM history_entries
                    )
                ''')
                
                # 清理未使用的标签
                cursor.execute('''
                    DELETE FROM tag_stats 
                    WHERE last_used < ?
                ''', (cutoff_date.isoformat(),))
                
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old history entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old entries: {str(e)}", exc_info=True)
            return 0

class HistoryHelpers:
    """历史记录辅助函数"""
    
    @staticmethod
    def create_conversation_entry(session_id: str, user_input: str, 
                                agent_response: str, user_id: str = None) -> HistoryEntry:
        """创建对话历史记录"""
        return HistoryEntry(
            entry_id=f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.datetime.now(),
            history_type=HistoryType.CONVERSATION,
            severity=EventSeverity.INFO,
            title="用户对话",
            description=f"用户: {user_input[:100]}...\nAgent: {agent_response[:100]}...",
            metadata={
                "user_input": user_input,
                "agent_response": agent_response,
                "input_length": len(user_input),
                "response_length": len(agent_response)
            },
            tags=["conversation", "user_interaction"],
            session_id=session_id,
            user_id=user_id
        )
    
    @staticmethod
    def create_execution_entry(session_id: str, execution_id: str,
                             code: str, success: bool, duration_ms: int,
                             stdout: str = "", stderr: str = "",
                             user_id: str = None) -> HistoryEntry:
        """创建代码执行历史记录"""
        return HistoryEntry(
            entry_id=f"exec_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.datetime.now(),
            history_type=HistoryType.CODE_EXECUTION,
            severity=EventSeverity.INFO if success else EventSeverity.ERROR,
            title=f"代码执行 - {'成功' if success else '失败'}",
            description=f"执行时长: {duration_ms}ms\n代码: {code[:200]}...",
            metadata={
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
                "code_length": len(code)
            },
            tags=["execution", "code", "success" if success else "error"],
            session_id=session_id,
            user_id=user_id,
            execution_id=execution_id,
            duration_ms=duration_ms,
            success=success
        )
    
    @staticmethod
    def create_ml_result_entry(session_id: str, execution_id: str,
                             result_data: Dict[str, Any], result_type: str,
                             user_id: str = None) -> HistoryEntry:
        """创建ML结果历史记录"""
        return HistoryEntry(
            entry_id=f"mlres_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.datetime.now(),
            history_type=HistoryType.ML_RESULT,
            severity=EventSeverity.INFO,
            title=f"ML结果保存 - {result_type}",
            description=f"保存了{result_type}类型的机器学习结果",
            metadata={
                "result_data": result_data,
                "result_type": result_type,
                "metrics_count": len(result_data) if isinstance(result_data, dict) else 0
            },
            tags=["ml_result", result_type, "machine_learning"],
            session_id=session_id,
            user_id=user_id,
            execution_id=execution_id
        )
    
    @staticmethod
    def create_error_entry(session_id: str, error_message: str,
                          error_type: str = "unknown", context: Dict[str, Any] = None,
                          user_id: str = None) -> HistoryEntry:
        """创建错误历史记录"""
        return HistoryEntry(
            entry_id=f"err_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.datetime.now(),
            history_type=HistoryType.ERROR_EVENT,
            severity=EventSeverity.ERROR,
            title=f"错误 - {error_type}",
            description=error_message,
            metadata=context or {},
            tags=["error", error_type],
            session_id=session_id,
            user_id=user_id,
            success=False
        )