#db_tools.py
import json
import datetime
import logging
from wsgiref.validate import validator
from MySQLdb import IntegrityError
from mysql.connector import Error as MySQLError
from typing import Dict, Any, Optional
from config import ExecutionErrorType # 从config导入
from db_utils import get_db_connection
from pydantic import BaseModel, Field, field_validator
from config import (
        GOOGLE_API_KEY,
        SYSTEM_PROMPT, 
        EXECUTE_CODE_TOOL_DESCRIPTION,
        SYSTEM_PROMPT_2,
        QUERY_EXEC_LOGS_TOOL_DESCRIPTION,
        QUERY_ML_RESULTS_TOOL_DESCRIPTION,
        SAVE_ML_RESULT_TOOL_DESCRIPTION,
        HOST_EXECUTION_LOGS_ROOT
)
import os
# --- 日志记录器设置 ---
# 确保日志记录器在模块级别配置一次
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # 防止重复添加处理器，如果此模块被多次导入
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 可以从config.py获取日志级别
    # from config import LOG_LEVEL
    # logger.setLevel(LOG_LEVEL)
    logger.setLevel(logging.INFO) # 或者先设为INFO，方便调试


# --- Pydantic 输入模型 (简化版，与config.py中的描述对应) ---

class DatabaseQueryInput(BaseModel):
    """用于查询系统执行日志的输入参数模型。"""
    query_description: str = Field(
        description="对你想要查询的系统执行日志的自然语言描述。"
    )
    execution_id_filter: Optional[str] = Field(
        default=None, 
        description="（可选）按执行ID精确查找。"
    )
    script_filename_contains: Optional[str] = Field(
        default=None, 
        description="（可选）筛选脚本文件名包含指定文本的记录 (部分匹配)。"
    )
    only_successful_runs: Optional[bool] = Field(
        default=None, 
        description="（可选）如果为True，只返回成功执行的记录；如果为False，只返回失败的记录；如果为None，则不按成功状态筛选。"
    )
    error_type_filter: Optional[ExecutionErrorType] = Field(
        default=None,
        description=f"（可选）按特定的错误类型筛选。有效值: {[e.value for e in ExecutionErrorType]}。"
    )
    limit: int = Field(
        default=5, ge=1, le=20, 
        description="返回的最大记录数。默认为5，最小1，最大20。"
    )

    @field_validator('error_type_filter', mode='before')
    def _validate_error_type_from_string(cls, value: Optional[str]) -> Optional[ExecutionErrorType]:
        """将字符串形式的error_type转换为ExecutionErrorType枚举实例。"""
        if value is None:
            return None
        if isinstance(value, ExecutionErrorType): # 如果已经是枚举实例
            return value
        try:
            return ExecutionErrorType(value)
        except ValueError:
            valid_values = [e.value for e in ExecutionErrorType]
            raise ValueError(f"无效的错误类型: '{value}'. 允许的值为: {valid_values}")


class SaveMLResultInput(BaseModel):
    """用于保存机器学习成果的输入参数模型。"""
    execution_id: str = Field(
        description="必需。产生这些机器学习结果的原始代码执行的唯一ID。"
    )
    result_data: Dict[str, Any] = Field(
        description="必需。一个包含你提取和结构化的机器学习结果的JSON对象（Python字典）。"
    )
    result_type: Optional[str] = Field(
        default=None,
        max_length=100, # 增加最大长度限制
        description="（可选）你为这组机器学习结果定义的类型或类别 (例如 'model_evaluation_metrics')。"
    )
    result_name: Optional[str] = Field(
        default=None,
        max_length=255, # 增加最大长度限制
        description="（可选）你为这组具体结果起的独特名称或标识 (例如 'Experiment_XGBoost_Run005_Eval')。"
    )
    ai_analysis_notes: Optional[str] = Field(
        default=None,
        description="（可选）你对这组结果的任何额外文字分析、解释、观察等。"
    )

class QueryMLResultsInput(BaseModel):
    """用于查询已保存机器学习成果的输入参数模型。"""
    query_description: str = Field(
        description="对你想要查询的已保存ML成果的自然语言描述。"
    )
    execution_id_filter: Optional[str] = Field(
        default=None,
        description="（可选）按关联的执行ID精确查找。"
    )
    result_type_contains: Optional[str] = Field(
        default=None,
        description="（可选）筛选成果类型 (`result_type`) 中包含指定文本的记录 (部分匹配)。"
    )
    result_name_contains: Optional[str] = Field(
        default=None,
        description="（可选）筛选成果名称 (`result_name`) 中包含指定文本的记录 (部分匹配)。"
    )
    limit: int = Field(
        default=3, ge=1, le=10,
        description="返回的最大ML成果记录条数。默认为3，最小1，最大10。"
    )

# --- 后端数据库操作函数 ---

def log_execution_run_to_db(execution_data: Dict[str, Any]) -> Optional[int]:
    """
    将单次代码执行的完整元数据记录到数据库的 execution_runs 表。
    这是 execute_ml_code_in_docker 函数在执行完毕后应该调用的核心日志记录函数。

    参数:
        execution_data (Dict[str, Any]): 一个包含执行元数据的字典，
                                         通常是 execute_ml_code_in_docker 返回的 result 字典。
                                         此字典的键应与 execution_runs 表的列名（或其子集）匹配。
    返回:
        新插入记录的ID (如果成功)，否则返回 None。
    """
    conn = None
    cursor = None
    inserted_id = None

    # 1. 准备要插入数据库的参数字典
    #    从 execution_data 中提取字段，并确保类型正确或提供默认值。
    #    字段名应与数据库表 `execution_runs` 的列名匹配。
    db_params = {
        "execution_id": execution_data.get("execution_id"),
        "timestamp_utc_start": execution_data.get("timestamp_utc_start"), # 期望是ISO格式字符串
        "timestamp_utc_end_process": execution_data.get("timestamp_utc_end_process"), # 期望是ISO格式字符串
        "script_filename_by_ai": execution_data.get("script_filename_by_ai"),
        "script_relative_path_by_ai": execution_data.get("script_relative_path_by_ai"),
        "ai_code_description": execution_data.get("ai_code_description"),
        "ai_code_purpose": execution_data.get("ai_code_purpose"),
        "use_gpu_requested": bool(execution_data.get("use_gpu_requested", False)),
        "timeout_seconds_set": execution_data.get("timeout_seconds_set"),
        # "cpu_core_limit_set": execution_data.get("cpu_core_limit_set"), # 如果表中没有这些列，则不应包含
        # "memory_limit_set": execution_data.get("memory_limit_set"),   # 同上

        "success": bool(execution_data.get("success", False)),
        "exit_code": execution_data.get("exit_code"),
        "execution_duration_seconds": execution_data.get("execution_duration_seconds"),
        "total_tool_duration_seconds": execution_data.get("total_tool_duration_seconds"),
        
        "error_type": None, # 先设为None，稍后处理枚举
        "error_message_preprocessing": execution_data.get("error_message_preprocessing"),
        "error_message_runtime": execution_data.get("error_message_runtime"),
        
        "log_directory_host_path": execution_data.get("log_directory_host_path"),
        "code_executed_host_path": execution_data.get("code_executed_host_path"),
        "executed_script_container_path": execution_data.get("executed_script_container_path"),
        "stdout_log_file_host_path": execution_data.get("stdout_log_file_host_path"),
        "stderr_log_file_host_path": execution_data.get("stderr_log_file_host_path"),
        "metadata_file_host_path": execution_data.get("metadata_file_host_path"),
        
        "stdout_summary": (execution_data.get("stdout", "") or "")[:1000], # 取前1000字符
        "stderr_summary": (execution_data.get("stderr", "") or "")[:1000], # 取前1000字符
    }

    # 处理 error_type 枚举 (如果存在且是枚举实例)
    error_type_value = execution_data.get("error_type")
    if isinstance(error_type_value, ExecutionErrorType):
        db_params["error_type"] = error_type_value.value
    elif isinstance(error_type_value, str) and error_type_value in [e.value for e in ExecutionErrorType]:
        db_params["error_type"] = error_type_value
    elif error_type_value is not None: # 如果是其他非None值，记录警告，并设为None或特定值
        logger.warning(f"接收到未知的 error_type 值: {error_type_value} for execution_id: {db_params['execution_id']}. 将存为 NULL。")
        db_params["error_type"] = None


    # 2. 校验必需字段 (根据您的数据库表设计调整)
    required_db_fields = [
        "execution_id", "timestamp_utc_start", "script_filename_by_ai", "success", "exit_code"
    ]
    missing_fields = [field for field in required_db_fields if db_params.get(field) is None]
    if missing_fields:
        logger.error(f"记录执行日志到数据库失败：execution_data中缺少必需字段：{missing_fields} for execution_id: {db_params.get('execution_id')}")
        return None

    # 3. 构建SQL插入语句 (动态生成列名和占位符，以适应db_params中实际存在的键)
    #    这要求db_params中的键名与数据库列名完全一致。
    columns_to_insert = [col for col, val in db_params.items() if val is not None] # 只插入非None的值
    if not columns_to_insert:
        logger.error(f"没有有效的字段可供插入数据库 for execution_id: {db_params.get('execution_id')}")
        return None
        
    sql_columns = ", ".join(columns_to_insert)
    sql_placeholders = ", ".join([f"%({col})s" for col in columns_to_insert])
    sql = f"INSERT INTO execution_runs ({sql_columns}) VALUES ({sql_placeholders})"
    
    # 准备实际传递给execute的参数字典，只包含非None的值
    final_params_for_sql = {col: db_params[col] for col in columns_to_insert}

    # 4. 执行数据库操作
    try:
        conn = get_db_connection()
        if not conn:
            logger.error(f"无法获取数据库连接以记录 execution_id: {db_params['execution_id']}")
            return None 
            
        cursor = conn.cursor()
        logger.debug(f"准备执行SQL: {sql} 参数: {final_params_for_sql}")
        cursor.execute(sql, final_params_for_sql)
        conn.commit()
        inserted_id = cursor.lastrowid
        logger.info(f"执行日志 (ID: {inserted_id}, ExecutionID: {db_params['execution_id']}) 已成功记录到数据库。")
        return inserted_id
            
    except MySQLError as e:
        logger.error(f"记录执行日志到MySQL时发生错误 (ExecutionID: {db_params.get('execution_id', 'N/A')}): {e}", exc_info=True)
        if conn:
            try: conn.rollback()
            except MySQLError as rb_e: logger.error(f"数据库回滚失败: {rb_e}", exc_info=True)
        return None
    except Exception as e: # 捕获其他所有可能的意外错误
        logger.error(f"记录执行日志时发生未知错误 (ExecutionID: {db_params.get('execution_id', 'N/A')}): {e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


def query_execution_history(
    query_description: str,
    execution_id_filter: Optional[str] = None,
    script_filename_contains: Optional[str] = None,
    only_successful_runs: Optional[bool] = None,
    error_type_filter: Optional[ExecutionErrorType] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    从 execution_runs 表查询代码执行的历史系统级日志 (简化版)。
    总是按执行开始时间降序排列 (最新的在前)。
    """
    conn = None
    cursor = None
    
    # 参数校验
    limit = max(1, min(limit, 20)) # 强制范围

    # 定义要选择的列 (与log_execution_run_to_db中插入的列对应，或其子集)
    select_columns_list = [
        "id", "execution_id", "timestamp_utc_start", "timestamp_utc_end_process", 
        "script_filename_by_ai", "script_relative_path_by_ai",
        "ai_code_description", "ai_code_purpose", "use_gpu_requested",
        "success", "exit_code", "execution_duration_seconds", "total_tool_duration_seconds",
        "error_type", "error_message_preprocessing", "error_message_runtime",
        "log_directory_host_path", "code_executed_host_path", 
        "stdout_log_file_host_path", "stderr_log_file_host_path", "metadata_file_host_path",
        "stdout_summary", "stderr_summary", "created_at" 
    ]
    select_columns_str = ", ".join(select_columns_list)

    base_sql = f"SELECT {select_columns_str} FROM execution_runs"
    where_clauses = []
    query_params_for_sql = {} 

    if execution_id_filter:
        where_clauses.append("execution_id = %(exec_id_param)s") # 使用不同的占位符名避免冲突
        query_params_for_sql["exec_id_param"] = execution_id_filter
    if script_filename_contains:
        where_clauses.append("script_filename_by_ai LIKE %(script_file_param)s")
        query_params_for_sql["script_file_param"] = f"%{script_filename_contains}%"
    if only_successful_runs is not None:
        where_clauses.append("success = %(success_param)s")
        query_params_for_sql["success_param"] = bool(only_successful_runs)
    if error_type_filter: # error_type_filter 已经是枚举实例
        where_clauses.append("error_type = %(error_type_param)s")
        query_params_for_sql["error_type_param"] = error_type_filter.value


    if where_clauses:
        base_sql += " WHERE " + " AND ".join(where_clauses)
        
    base_sql += " ORDER BY timestamp_utc_start DESC"
    base_sql += " LIMIT %(limit_param)s"
    query_params_for_sql["limit_param"] = limit

    logger.info(f"AI查询执行历史意图: '{query_description}'")
    logger.debug(f"执行SQL查询 execution_runs: {base_sql} (参数: {query_params_for_sql})")

    try:
        conn = get_db_connection()
        if not conn: return {"error": "数据库连接失败。"}
            
        cursor = conn.cursor(dictionary=True) # 返回字典形式的结果
        cursor.execute(base_sql, query_params_for_sql)
        fetched_results = cursor.fetchall()
            
        # 将datetime对象转换为ISO格式字符串以便JSON序列化
        processed_results = []
        for row in fetched_results:
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, datetime.datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            processed_results.append(processed_row)
            
        logger.info(f"执行历史查询返回 {len(processed_results)} 条记录。")
        return {
            "results": processed_results, 
            "query_details": {
                "description_from_ai": query_description,
                "filters_applied": {
                    "execution_id": execution_id_filter,
                    "script_filename_contains": script_filename_contains,
                    "only_successful_runs": only_successful_runs,
                    "error_type": error_type_filter.value if error_type_filter else None,
                    "limit": limit
                },
                "sql_approx": base_sql # 近似的SQL（占位符未替换）
            }
        }
            
    except MySQLError as e:
        logger.error(f"查询执行历史时发生数据库错误: {e}", exc_info=True)
        return {"error": f"数据库查询失败: {e}"}
    except Exception as e:
        logger.error(f"查询执行历史时发生未知错误: {e}", exc_info=True)
        return {"error": f"未知错误: {e}"}
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


def save_ml_result_to_db(
    execution_id: str,
    result_data: Dict[str, Any],
    result_type: Optional[str] = None,
    result_name: Optional[str] = None,
    ai_analysis_notes: Optional[str] = None
) -> Dict[str, Any]:
    """将AI分析和提取的机器学习结果以JSON格式保存到 ml_results 表。"""
    conn = None
    cursor = None
    
    # 0. 验证 execution_id 是否有效 (在 execution_runs 表中存在)
    conn_check = None
    cursor_check = None
    try:
        conn_check = get_db_connection()
        if not conn_check:
            return {"success": False, "error_type": "DatabaseConnectionError", "message": "无法连接数据库进行execution_id校验。"}
        cursor_check = conn_check.cursor(dictionary=True) # 使用dictionary=True方便获取列
        cursor_check.execute("SELECT id, execution_id FROM execution_runs WHERE execution_id = %s LIMIT 1", (execution_id,))
        run_record = cursor_check.fetchone()
        if not run_record:
            error_msg = f"提供的 execution_id '{execution_id}' 在 execution_runs 表中不存在或无效。"
            logger.warning(error_msg)
            return {"success": False, "error_type": "InvalidExecutionID", "message": error_msg}
        # (可选) 可以将查到的 execution_runs.id (主键) 保存到 ml_results 表的 execution_run_db_id 列（如果设计了这样的列）
        # execution_run_db_id = run_record['id'] 
    except MySQLError as e_check:
        error_msg = f"校验 execution_id '{execution_id}' 时发生数据库错误: {e_check}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error_type": "DatabaseValidationError", "message": error_msg}
    finally:
        if cursor_check: cursor_check.close()
        if conn_check and conn_check.is_connected(): conn_check.close()

    # 1. 序列化 result_data
    try:
        # indent=None 使其更紧凑，如果不需要在数据库中直接阅读格式化的JSON
        result_data_json_str = json.dumps(result_data, ensure_ascii=False, indent=None) 
    except TypeError as e:
        error_msg = f"传递给 'result_data' 的内容无法序列化为JSON: {e}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error_type": "InvalidInputFormat", "message": error_msg}

    current_utc_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 2. 准备SQL语句和参数
    # 注意：表名和列名应与您的数据库模式完全匹配
    sql = """
    INSERT INTO ml_results (
        execution_run_id, result_timestamp_utc, result_type, 
        result_name, result_data_json, ai_analysis_notes
    ) VALUES (
        %(execution_id_param)s, %(result_timestamp_param)s, %(result_type_param)s,
        %(result_name_param)s, %(result_data_json_param)s, %(ai_analysis_notes_param)s
    )
    """
    params_to_insert = {
        "execution_id_param": execution_id,
        "result_timestamp_param": current_utc_time_iso,
        "result_type_param": result_type,
        "result_name_param": result_name,
        "result_data_json_param": result_data_json_str,
        "ai_analysis_notes_param": ai_analysis_notes
    }

    # 3. 执行数据库插入操作
    try:
        conn = get_db_connection()
        if not conn:
            return {"success": False, "error_type": "DatabaseConnectionError", "message": "无法连接到数据库保存ML成果。"}
            
        cursor = conn.cursor()
        cursor.execute(sql, params_to_insert)
        conn.commit()
        ml_result_id = cursor.lastrowid
        
        success_msg = f"机器学习成果已成功保存。关联执行ID: '{execution_id}', 新ML成果记录ID: {ml_result_id}."
        logger.info(success_msg)
        return {"success": True, "ml_result_id": ml_result_id, "message": success_msg}
            
    except IntegrityError as ie:
        error_msg = f"保存ML成果时发生数据库完整性错误 (execution_id '{execution_id}' 可能仍有问题或数据格式不匹配): {ie}"
        logger.error(error_msg, exc_info=True)
        if conn: conn.rollback()
        return {"success": False, "error_type": "DatabaseIntegrityError", "message": error_msg}
    except MySQLError as e:
        error_msg = f"保存ML成果到MySQL时发生错误 (关联执行ID: '{execution_id}'): {e}"
        logger.error(error_msg, exc_info=True)
        if conn: conn.rollback()
        # 尝试提取MySQL错误码
        mysql_error_code = f"MySQL_{e.errno}" if hasattr(e, 'errno') else "DatabaseError"
        return {"success": False, "error_type": mysql_error_code, "message": error_msg}
    except Exception as e: # 捕获其他所有可能的意外错误
        error_msg = f"保存ML成果时发生未知错误 (关联执行ID: '{execution_id}'): {e}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error_type": "UnknownError", "message": error_msg}
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


def query_ml_results_from_db(
    query_description: str,
    execution_id_filter: Optional[str] = None,
    result_type_contains: Optional[str] = None,
    result_name_contains: Optional[str] = None,
    limit: int = 3
) -> Dict[str, Any]:
    """从 ml_results 表查询已保存的机器学习结果 (简化版)。总是按保存时间降序排列。"""
    conn = None
    cursor = None
    
    limit = max(1, min(limit, 10)) # 强制范围

    select_columns_str = "id, execution_run_id, result_timestamp_utc, result_type, result_name, result_data_json, ai_analysis_notes, created_at"
    base_sql = f"SELECT {select_columns_str} FROM ml_results"
    where_clauses = []
    query_params_for_sql = {}

    if execution_id_filter:
        where_clauses.append("execution_run_id = %(exec_id_param)s")
        query_params_for_sql["exec_id_param"] = execution_id_filter
    if result_type_contains:
        where_clauses.append("result_type LIKE %(res_type_param)s")
        query_params_for_sql["res_type_param"] = f"%{result_type_contains}%"
    if result_name_contains:
        where_clauses.append("result_name LIKE %(res_name_param)s")
        query_params_for_sql["res_name_param"] = f"%{result_name_contains}%"
    
    if where_clauses:
        base_sql += " WHERE " + " AND ".join(where_clauses)
        
    base_sql += " ORDER BY result_timestamp_utc DESC"
    base_sql += " LIMIT %(limit_param)s"
    query_params_for_sql["limit_param"] = limit

    logger.info(f"AI查询ML成果意图: '{query_description}'")
    logger.debug(f"执行SQL查询 ml_results: {base_sql} (参数: {query_params_for_sql})")

    try:
        conn = get_db_connection()
        if not conn: return {"error": "数据库连接失败。"}
            
        cursor = conn.cursor(dictionary=True)
        cursor.execute(base_sql, query_params_for_sql)
        fetched_results = cursor.fetchall()
            
        processed_results = []
        for row in fetched_results:
            processed_row = {}
            for key, value in row.items():
                if key == "result_data_json" and isinstance(value, str):
                    try: processed_row[key] = json.loads(value) # 将JSON字符串解析为Python对象
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析ml_result_id: {row.get('id')} 的result_data_json。返回原始字符串。")
                        processed_row[key] = value # 保留原始字符串
                elif isinstance(value, datetime.datetime):
                    processed_row[key] = value.isoformat() # datetime -> ISO string
                else:
                    processed_row[key] = value
            processed_results.append(processed_row)

        logger.info(f"ML成果查询返回 {len(processed_results)} 条记录。")
        return {
            "results": processed_results, 
            "query_details": {
                "description_from_ai": query_description,
                "filters_applied": {
                    "execution_id": execution_id_filter,
                    "result_type_contains": result_type_contains,
                    "result_name_contains": result_name_contains,
                    "limit": limit
                },
                "sql_approx": base_sql
            }
        }
            
    except MySQLError as e:
        logger.error(f"查询ML成果时发生数据库错误: {e}", exc_info=True)
        return {"error": f"数据库查询失败: {e}"}
    except Exception as e:
        logger.error(f"查询ML成果时发生未知错误: {e}", exc_info=True)
        return {"error": f"未知错误: {e}"}
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


# --- 主模块测试代码 (保持与之前类似，但现在调用的是db_tools.py中的函数) ---
if __name__ == '__main__':
    # 配置日志记录器以查看输出 (如果尚未在顶层配置)
    if not logging.getLogger().hasHandlers(): # 检查根logger是否已有处理器
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("开始数据库工具测试 (db_tools.py)...")

    # --- 测试 log_execution_run_to_db ---
    # 模拟 execute_ml_code_in_docker 的返回数据
    # 注意：确保 execution_id 是唯一的，或者您的表设计允许重复（不推荐）
    current_time_for_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    mock_execution_id = f"test_exec_{current_time_for_id}"
    
    mock_execution_data = {
        "execution_id": mock_execution_id,
        "timestamp_utc_start": datetime.datetime.utcnow().isoformat() + "Z",
        "timestamp_utc_end_process": (datetime.datetime.utcnow() + datetime.timedelta(seconds=10)).isoformat() + "Z",
        "script_filename_by_ai": "test_script_for_db.py",
        "script_relative_path_by_ai": "db_tests",
        "ai_code_description": "A test script for DB logging.",
        "ai_code_purpose": "Verify db_tools.log_execution_run_to_db.",
        "use_gpu_requested": False,
        "timeout_seconds_set": 60,
        "success": True,
        "exit_code": 0,
        "execution_duration_seconds": 5.123,
        "total_tool_duration_seconds": 6.789,
        "error_type": None, 
        "error_message_preprocessing": None,
        "error_message_runtime": None,
        "log_directory_host_path": os.path.join(HOST_EXECUTION_LOGS_ROOT, mock_execution_id),
        "code_executed_host_path": os.path.join(HOST_EXECUTION_LOGS_ROOT, mock_execution_id, "db_tests", "test_script_for_db.py"),
        "executed_script_container_path": "/sandbox/code/db_tests/test_script_for_db.py",
        "stdout_log_file_host_path": os.path.join(HOST_EXECUTION_LOGS_ROOT, mock_execution_id, f"{mock_execution_id}.stdout.log"),
        "stderr_log_file_host_path": os.path.join(HOST_EXECUTION_LOGS_ROOT, mock_execution_id, f"{mock_execution_id}.stderr.log"),
        "metadata_file_host_path": os.path.join(HOST_EXECUTION_LOGS_ROOT, mock_execution_id, f"{mock_execution_id}.meta.json"),
        "stdout": "Test stdout for DB log.",
        "stderr": ""
    }
    logger.info(f"\n--- 测试 log_execution_run_to_db ---")
    inserted_run_id = log_execution_run_to_db(mock_execution_data)
    if inserted_run_id:
        logger.info(f"成功记录执行日志到数据库，新记录ID: {inserted_run_id}")
    else:
        logger.error("记录执行日志到数据库失败。")
    
    # --- 测试 query_execution_history ---
    logger.info(f"\n--- 测试 query_execution_history (查询最近2条) ---")
    history_results = query_execution_history(
        query_description="获取最近的执行记录",
        limit=2
    )
    logger.info(f"查询执行历史结果: {json.dumps(history_results, indent=2, ensure_ascii=False)}")
    
    test_execution_id_for_ml_result = None
    if history_results and history_results.get("results") and len(history_results["results"]) > 0:
        # 确保我们使用的是刚刚插入的记录的execution_id（如果插入成功）
        # 或者至少是最近的一条记录
        if inserted_run_id: # 如果上一步插入成功
             # 再次查询以确保获取到的是我们刚插入的，因为可能有并发
            check_inserted = query_execution_history(query_description="check inserted", execution_id_filter=mock_execution_id, limit=1)
            if check_inserted and check_inserted.get("results"):
                test_execution_id_for_ml_result = check_inserted["results"][0]["execution_id"]
                logger.info(f"将使用 execution_id '{test_execution_id_for_ml_result}' 进行后续ML成果测试。")
            else:
                 logger.warning(f"未能通过 execution_id '{mock_execution_id}' 重新获取到刚插入的执行记录。")
        
        if not test_execution_id_for_ml_result: # 如果通过ID查找失败，则用列表中的第一个
            test_execution_id_for_ml_result = history_results["results"][0]["execution_id"]
            logger.info(f"将使用列表中的第一个 execution_id '{test_execution_id_for_ml_result}' 进行后续ML成果测试。")


        logger.info(f"\n--- 测试 query_execution_history (按execution_id查询) ---")
        specific_history = query_execution_history(
            query_description=f"查找特定执行 {test_execution_id_for_ml_result}",
            execution_id_filter=test_execution_id_for_ml_result,
            limit=1
        )
        logger.info(f"特定执行历史查询结果: {json.dumps(specific_history, indent=2, ensure_ascii=False)}")

        # --- 测试 save_ml_result_to_db ---
        if test_execution_id_for_ml_result: # 确保我们有一个有效的execution_id
            logger.info(f"\n--- 测试 save_ml_result_to_db (使用 execution_id: {test_execution_id_for_ml_result}) ---")
            mock_ml_data = {
                "model_type": "DecisionTreeClassifier",
                "accuracy_score": 0.91,
                "config": {"max_depth": 7, "min_samples_leaf": 5},
                "notes": "Good performance on validation set."
            }
            save_status = save_ml_result_to_db(
                execution_id=test_execution_id_for_ml_result,
                result_data=mock_ml_data,
                result_type="classification_model_metrics",
                result_name="DecisionTree_Run_Final_Eval",
                ai_analysis_notes="The model seems to generalize well. Next step: deploy."
            )
            logger.info(f"保存ML成果结果: {json.dumps(save_status, indent=2, ensure_ascii=False)}")
            
            if save_status and save_status.get("success"):
                # --- 测试 query_ml_results_from_db ---
                logger.info(f"\n--- 测试 query_ml_results_from_db (查询最近1条ML成果) ---")
                ml_query_res = query_ml_results_from_db(
                    query_description="获取最近保存的ML成果",
                    limit=1
                )
                logger.info(f"查询ML成果结果: {json.dumps(ml_query_res, indent=2, ensure_ascii=False)}")

                logger.info(f"\n--- 测试 query_ml_results_from_db (按类型查询) ---")
                ml_query_by_type = query_ml_results_from_db(
                    query_description="查找所有分类模型指标",
                    result_type_contains="classification_model_metrics", # 确保与上面保存的一致
                    limit=5
                )
                logger.info(f"按类型查询ML成果结果: {json.dumps(ml_query_by_type, indent=2, ensure_ascii=False)}")
            else:
                logger.warning("由于保存ML成果失败，跳过后续的ML成果查询测试。")
        else:
            logger.warning("没有有效的 execution_id 可用于测试 save_ml_result_to_db。")
    else:
        logger.warning("由于查询执行历史失败或无结果，跳过后续的ML成果相关测试。")

    logger.info("\n数据库工具测试结束 (db_tools.py)。")