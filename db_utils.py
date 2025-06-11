# db_utils.py
import mysql.connector
from mysql.connector import Error as MySQLError # 使用别名以避免冲突
from typing import Dict, Any, Optional
import logging
from config import DB_CONFIG # 从config导入

logger = logging.getLogger(__name__)

def get_db_connection() -> Optional[mysql.connector.MySQLConnection]:
    """建立并返回一个MySQL数据库连接。"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            logger.debug("成功连接到MySQL数据库。")
            return conn
        else:
            logger.error("连接MySQL失败：未建立连接。")
            return None
    except MySQLError as e:
        logger.error(f"连接MySQL数据库时出错：{e}", exc_info=True)
        return None

def log_execution_to_database(execution_data: Dict[str, Any]) -> Optional[int]:
    """
    将执行元数据记录到MySQL数据库。
    返回新插入记录的ID，失败时返回None。
    """
    conn = None
    cursor = None
    record_id = None

    # 准备插入的数据，确保所有键存在或有默认值
    # 此映射有助于确保我们只尝试插入表中存在的列
    # 并优雅地处理execution_data中缺失的键。
    db_payload = {
        "execution_id": execution_data.get("execution_id"),
        "timestamp_utc_start": execution_data.get("timestamp_utc_start"),
        "timestamp_utc_end_process": execution_data.get("timestamp_utc_end_process"),
        "script_filename_by_ai": execution_data.get("script_filename_by_ai"),
        "script_relative_path_by_ai": execution_data.get("script_relative_path_by_ai"),
        "ai_code_description": execution_data.get("ai_code_description"),
        "ai_code_purpose": execution_data.get("ai_code_purpose"),
        "use_gpu_requested": bool(execution_data.get("use_gpu_requested", False)),
        "timeout_seconds_set": execution_data.get("timeout_seconds_set"),
        "cpu_core_limit_set": execution_data.get("cpu_core_limit_set"),
        "memory_limit_set": execution_data.get("memory_limit_set"),
        "success": bool(execution_data.get("success", False)),
        "exit_code": execution_data.get("exit_code"),
        "execution_duration_seconds": execution_data.get("execution_duration_seconds"),
        "total_tool_duration_seconds": execution_data.get("total_tool_duration_seconds"),
        "error_type": execution_data.get("error_type"), # 假设error_type已经是Enum中的字符串值
        "error_message_preprocessing": execution_data.get("error_message_preprocessing"),
        "error_message_runtime": execution_data.get("error_message_runtime"),
        "log_directory_host_path": execution_data.get("log_directory_host_path"),
        "code_executed_host_path": execution_data.get("code_executed_host_path"),
        "stdout_log_file_host_path": execution_data.get("stdout_log_file_host_path"),
        "stderr_log_file_host_path": execution_data.get("stderr_log_file_host_path"),
        "metadata_file_host_path": execution_data.get("metadata_file_host_path"),
        "stdout_summary": (execution_data.get("stdout", "") or "")[:1000], # 示例摘要
        "stderr_summary": (execution_data.get("stderr", "") or "")[:1000], # 示例摘要
    }

    # 确保数据库所需的所有字段都存在，否则记录错误并跳过
    required_db_fields = [
        "execution_id", "timestamp_utc_start", "script_filename_by_ai", 
        "script_relative_path_by_ai", "use_gpu_requested", "timeout_seconds_set",
        "success", "exit_code", "log_directory_host_path", 
        "code_executed_host_path", "stdout_log_file_host_path",
        "stderr_log_file_host_path", "metadata_file_host_path"
    ]
    missing_fields = [field for field in required_db_fields if db_payload.get(field) is None]
    if missing_fields:
        logger.error(f"无法记录到数据库：execution_data中缺少必需字段：{missing_fields}")
        return None

    # 动态构建SQL查询以正确处理可选字段
    columns = []
    placeholders = []
    values_to_insert = []

    for key, value in db_payload.items():
        # 仅包含不为None的键，或根据数据库模式调整为NULL
        # 为简单起见，这里假设数据库列对于可选字段是可为空的
        # 或者db_payload为不可为空的字段提供了适当的默认值。
        columns.append(key)
        placeholders.append(f"%({key})s")
        values_to_insert.append(value)
        
    # 构建SQL查询
    # 如果有许多可选字段并希望避免发送NULL，这是一种更稳健的方法
    # 对于可能不在db_payload中或显式为None的列。
    # 然而，为简单起见，如果表允许可选字段为NULL，
    # 之前的静态SQL与完整的params_to_insert字典也是可以的。
    # 现在让我们坚持使用静态SQL，假设表处理NULL。

    sql = f"""
    INSERT INTO execution_runs ({', '.join(db_payload.keys())}) 
    VALUES ({', '.join([f'%({k})s' for k in db_payload.keys()])})
    """
    
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(sql, db_payload) # 传递整个字典
            conn.commit()
            record_id = cursor.lastrowid
            logger.info(f"执行记录（ID: {execution_data.get('execution_id')}）成功插入数据库，数据库ID: {record_id}")
        else:
            logger.warning("无法记录到数据库：没有数据库连接。")
            
    except MySQLError as e:
        logger.error(f"将执行记录（ID: {execution_data.get('execution_id')}）插入MySQL时出错：{e}", exc_info=True)
        if conn:
            conn.rollback()
    except Exception as e: # 捕获数据库交互期间的其他潜在错误
        logger.error(f"在数据库记录期间执行ID {execution_data.get('execution_id')}时发生意外错误：{e}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
    
    return record_id