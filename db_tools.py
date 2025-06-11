#db_tools.py
import json
import datetime
import logging
from mysql.connector import Error as MySQLError
from typing import Dict, Any, Optional
from config import ExecutionErrorType # 从config导入
from db_utils import get_db_connection
from pydantic import BaseModel, Field
logger = logging.getLogger(__name__) # 确保logger已定义

class DatabaseQueryInput(BaseModel):
    query_description: str = Field(
        description="对你想要从执行历史数据库中查询的系统级日志信息的自然语言描述。例如：'查找最近5次失败的执行记录' 或 '显示所有脚本名为 train_model.py 且成功的执行'。"
    )
    execution_id_filter: Optional[str] = Field(
        default=None, description="（可选）按特定的执行ID精确查找。"
    )
    script_filename_filter: Optional[str] = Field(
        default=None, description="（可选）按AI指定的脚本文件名筛选 (例如 'train_model.py')。支持部分匹配。"
    )
    success_status_filter: Optional[bool] = Field(
        default=None, description="（可选）按执行成功状态筛选 (True 表示成功, False 表示失败)。"
    )
    error_type_filter: Optional[str] = Field(
        default=None, description=f"（可选）按特定的错误类型筛选。有效值: {[e.value for e in ExecutionErrorType]}。"
    )
    description_contains_filter: Optional[str] = Field(
        default=None, description="（可选）筛选AI代码描述 (ai_code_description) 中包含指定文本的记录。"
    )
    purpose_contains_filter: Optional[str] = Field(
        default=None, description="（可选）筛选AI代码目的 (ai_code_purpose) 中包含指定文本的记录。"
    )
    limit: int = Field(
        default=10, ge=1, le=50,
        description="返回的最大执行日志记录条数。默认为10，最小1，最大50。"
    )
    offset: int = Field(
        default=0, ge=0,
        description="用于分页，跳过前面多少条记录。默认为0。"
    )
    sort_by_column: Optional[str] = Field(
        default="timestamp_utc_start",
        description="（可选）指定用于排序结果的列名。允许的列包括：'id', 'execution_id', 'timestamp_utc_start', 'script_filename_by_ai', 'success', 'exit_code', 'execution_duration_seconds', 'error_type'。默认为'timestamp_utc_start'。"
    )
    sort_order: str = Field(
        default="DESC", pattern="^(ASC|DESC|asc|desc)$",
        description="（可选）排序顺序。可以是 'ASC' (升序) 或 'DESC' (降序)。默认为'DESC' (最新的在前)。"
    )

class SaveMLResultInput(BaseModel):
    execution_id: str = Field(
        description="必需。产生这些机器学习结果的原始代码执行的唯一ID。这个ID通常可以从 ExecutePythonInMLSandbox 工具的成功返回中获取，或者通过 QuerySystemExecutionLogs 工具查询得到。"
    )
    result_data: Dict[str, Any] = Field(
        description="必需。一个包含你提取和结构化的机器学习结果的JSON对象（在Python中表现为字典）。这个字典的结构完全由你根据当前任务和结果的性质来定义。力求清晰、信息完整且易于后续解析。例如：{'accuracy': 0.95, 'metrics_set': {'precision': 0.92, 'recall': 0.98}} 或 {'top_features': ['age', 'income'], 'scores': [0.8, 0.75]}。"
    )
    result_type: Optional[str] = Field(
        default=None,
        description="（可选）你为这组机器学习结果定义的类型或类别，方便后续筛选和理解。例如：'model_evaluation_metrics', 'feature_importance_scores', 'data_summary_statistics', 'hyperparameter_tuning_result', 'anomaly_detection_report'。"
    )
    result_name: Optional[str] = Field(
        default=None,
        description="（可选）你为这组具体结果起的独特名称或标识，使其易于识别和引用。例如：'Experiment_XGBoost_Run005_ValidationSet_Eval', 'CustomerChurn_FeatureImportances_V2', 'SalesData_Q1_2025_Summary'。"
    )
    ai_analysis_notes: Optional[str] = Field(
        default=None,
        description="（可选）你对这组结果的任何额外文字分析、解释、观察、遇到的挑战、或下一步的建议。这可以是对 result_data 的补充说明。"
    )
    # result_timestamp_utc 将在后端自动生成并记录，AI无需提供。

class QueryMLResultsInput(BaseModel):
    query_description: str = Field(
        description="对你想要从已保存的机器学习结果中查询的信息的自然语言描述。例如：'查找所有模型评估结果' 或 '显示与特定执行ID关联的特征重要性得分'。"
    )
    # --- 筛选条件 ---
    execution_id_filter: Optional[str] = Field(
        default=None,
        description="（可选）按产生该ML结果的原始代码执行的ID (execution_id) 进行精确筛选。"
    )
    result_type_filter: Optional[str] = Field(
        default=None,
        description="（可选）按ML结果的类型 (result_type) 进行部分匹配筛选。例如：'model_evaluation', 'feature_importance'。"
    )
    result_name_filter: Optional[str] = Field(
        default=None,
        description="（可选）按ML结果的名称 (result_name) 进行部分匹配筛选。例如：'ResNet50_Accuracy', 'XGBoost_Run003'。"
    )
    # --- JSON内容筛选 (高级，初期可简化或省略) ---
    # json_contains_key_filter: Optional[str] = Field(
    #     default=None,
    #     description="（可选）筛选 result_data_json 中存在特定顶级键的记录。"
    # )
    # json_path_value_filter: Optional[Dict[str, Any]] = Field(
    #     default=None,
    #     description="（可选）一个字典，键是JSON路径表达式 (例如 '$.metrics.accuracy')，值是期望的值，用于筛选 result_data_json。"
    # ) # 这个实现起来会比较复杂且有安全风险，初期强烈建议不直接暴露给AI

    # --- 控制返回结果 ---
    limit: int = Field(
        default=5, ge=1, le=30,
        description="返回的最大机器学习结果记录条数。默认为5，最小1，最大30。"
    )
    offset: int = Field(
        default=0, ge=0,
        description="用于分页，跳过前面多少条记录。默认为0。"
    )
    sort_by_column: Optional[str] = Field(
        default="result_timestamp_utc",
        description="（可选）指定用于排序结果的列名。允许的列包括：'id', 'execution_run_id', 'result_timestamp_utc', 'result_type', 'result_name'。默认为'result_timestamp_utc'。"
    )
    sort_order: str = Field(
        default="DESC",
        pattern="^(ASC|DESC|asc|desc)$", # 校验值
        description="（可选）排序顺序。可以是 'ASC' (升序) 或 'DESC' (降序)。默认为'DESC' (最新的在前)。"
    )

def query_execution_history(
    # 参数与 DatabaseQueryInput 中的字段名完全对应
    query_description: str, # 虽然主要用于AI组织意图，但函数接收它以保持一致性
    execution_id_filter: Optional[str] = None,
    script_filename_filter: Optional[str] = None,
    success_status_filter: Optional[bool] = None,
    error_type_filter: Optional[str] = None,
    description_contains_filter: Optional[str] = None,
    purpose_contains_filter: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    sort_by_column: str = "timestamp_utc_start", # Pydantic模型有Optional，但函数这里可以设为str并有默认
    sort_order: str = "DESC"
) -> Dict[str, Any]:
    """
    从 execution_runs 表查询代码执行的历史系统级日志。
    接收的参数直接来自 DatabaseQueryInput Pydantic 模型。
    """
    conn = None
    cursor = None
    
    # 参数校验和清理
    limit = max(1, min(limit, 50)) 
    offset = max(0, offset)
    
    allowed_sort_columns = [ # 与 execution_runs 表的列对应
        "id", "execution_id", "timestamp_utc_start", "timestamp_utc_end_process",
        "script_filename_by_ai", "success", "exit_code", "execution_duration_seconds",
        "total_tool_duration_seconds", "error_type", "created_at" 
    ]
    if sort_by_column not in allowed_sort_columns:
        logger.warning(f"Invalid sort_by_column for execution_runs: {sort_by_column}. Defaulting to 'timestamp_utc_start'.")
        sort_by_column = "timestamp_utc_start"
        
    if sort_order.upper() not in ["ASC", "DESC"]:
        logger.warning(f"Invalid sort_order for execution_runs: {sort_order}. Defaulting to 'DESC'.")
        sort_order = "DESC"

    # 定义要从 execution_runs 表返回的核心列
    # 避免选择非常大的TEXT字段如完整的stdout/stderr，那些通过文件路径引用
    select_columns_str = """
        id, execution_id, timestamp_utc_start, timestamp_utc_end_process, 
        script_filename_by_ai, script_relative_path_by_ai,
        ai_code_description, ai_code_purpose, use_gpu_requested,
        success, exit_code, execution_duration_seconds, total_tool_duration_seconds,
        error_type, error_message_preprocessing, error_message_runtime,
        log_directory_host_path, code_executed_host_path, 
        stdout_log_file_host_path, stderr_log_file_host_path, metadata_file_host_path,
        stdout_summary, stderr_summary, created_at 
    """
    # (可以根据AI实际需要调整这里的列，但要避免返回过多不必要的数据)

    base_sql = f"SELECT {select_columns_str} FROM execution_runs"
    where_clauses = []
    query_params = {} # 用于参数化查询

    if execution_id_filter:
        where_clauses.append("execution_id = %(execution_id)s")
        query_params["execution_id"] = execution_id_filter
    if script_filename_filter:
        where_clauses.append("script_filename_by_ai LIKE %(script_filename)s")
        query_params["script_filename"] = f"%{script_filename_filter}%"
    if success_status_filter is not None:
        where_clauses.append("success = %(success_status)s")
        query_params["success_status"] = bool(success_status_filter) # 确保是布尔值
    if error_type_filter:
        valid_error_types = [e.value for e in ExecutionErrorType]
        if error_type_filter in valid_error_types:
            where_clauses.append("error_type = %(error_type)s")
            query_params["error_type"] = error_type_filter
        else:
            logger.warning(f"Invalid error_type_filter for execution_runs: {error_type_filter}. Ignoring.")
    if description_contains_filter:
        where_clauses.append("ai_code_description LIKE %(description_contains)s")
        query_params["description_contains"] = f"%{description_contains_filter}%"
    if purpose_contains_filter:
        where_clauses.append("ai_code_purpose LIKE %(purpose_contains)s")
        query_params["purpose_contains"] = f"%{purpose_contains_filter}%"

    if where_clauses:
        base_sql += " WHERE " + " AND ".join(where_clauses)
        
    base_sql += f" ORDER BY {sort_by_column} {sort_order.upper()}" # sort_by_column已白名单校验
    base_sql += " LIMIT %(limit)s OFFSET %(offset)s"
    query_params["limit"] = limit
    query_params["offset"] = offset

    logger.debug(f"Executing Execution History query: {base_sql} with params: {query_params}")
    # logger.info(f"Query description from AI: {query_description}") # 可以记录AI的意图

    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(base_sql, query_params)
            results = cursor.fetchall()
            
            # 将datetime对象转换为ISO格式字符串
            for row in results:
                for key, value in row.items():
                    if isinstance(value, datetime.datetime):
                        row[key] = value.isoformat()
            
            logger.info(f"Execution History query returned {len(results)} records.")
            # 可以在这里添加获取总匹配数（不带limit/offset）的逻辑，如果需要
            # total_matches_sql = f"SELECT COUNT(*) as total FROM execution_runs {('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''}"
            # cursor.execute(total_matches_sql, {k:v for k,v in query_params.items() if k not in ['limit', 'offset']}) # 移除limit/offset参数
            # total_matches = cursor.fetchone()['total']

            return {
                "results": results, 
                # "total_matches": total_matches, # (可选)
                "query_executed_approx": base_sql.replace("%(limit)s", str(limit)).replace("%(offset)s", str(offset)) # 调试用
            }
        else:
            return {"error": "Failed to connect to the database for querying execution history."}
            
    except MySQLError as e:
        logger.error(f"Error querying execution history: {e}", exc_info=True)
        return {"error": f"Database query for execution history failed: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error during execution history query: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during execution history query: {e}"}
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def query_ml_results_from_db(
    execution_id_filter: Optional[str] = None,
    result_type_filter: Optional[str] = None,
    result_name_filter: Optional[str] = None,
    limit: int = 5, # 与Pydantic模型默认值一致
    offset: int = 0,
    sort_by_column: str = "result_timestamp_utc", # 与Pydantic模型默认值一致
    sort_order: str = "DESC" # 与Pydantic模型默认值一致
) -> Dict[str, Any]:
    """
    从 ml_results 表查询已保存的机器学习结果。

    返回:
        一个字典，包含 "results" (一个ML结果记录列表) 或 "error"。
    """
    conn = None
    cursor = None
    
    # 参数校验和清理
    limit = max(1, min(limit, 30)) # 强制限制
    offset = max(0, offset)
    
    allowed_sort_columns = [
        "id", "execution_run_id", "result_timestamp_utc", 
        "result_type", "result_name", "created_at"
    ]
    if sort_by_column not in allowed_sort_columns:
        logger.warning(f"Invalid sort_by_column for ml_results: {sort_by_column}. Defaulting to 'result_timestamp_utc'.")
        sort_by_column = "result_timestamp_utc"
        
    if sort_order.upper() not in ["ASC", "DESC"]:
        logger.warning(f"Invalid sort_order for ml_results: {sort_order}. Defaulting to 'DESC'.")
        sort_order = "DESC"

    # 返回所有列，因为result_data_json是核心
    select_columns_str = "id, execution_run_id, result_timestamp_utc, result_type, result_name, result_data_json, ai_analysis_notes, created_at"

    base_sql = f"SELECT {select_columns_str} FROM ml_results"
    where_clauses = []
    query_params = {}

    if execution_id_filter:
        where_clauses.append("execution_run_id = %(exec_id)s")
        query_params["exec_id"] = execution_id_filter
    if result_type_filter:
        where_clauses.append("result_type LIKE %(res_type)s")
        query_params["res_type"] = f"%{result_type_filter}%"
    if result_name_filter:
        where_clauses.append("result_name LIKE %(res_name)s")
        query_params["res_name"] = f"%{result_name_filter}%"
    
    # 注意：我们没有实现复杂的JSON内部查询，AI获取整个JSON后自行解析

    if where_clauses:
        base_sql += " WHERE " + " AND ".join(where_clauses)
        
    base_sql += f" ORDER BY {sort_by_column} {sort_order.upper()}"
    base_sql += " LIMIT %(limit)s OFFSET %(offset)s"
    query_params["limit"] = limit
    query_params["offset"] = offset

    logger.debug(f"Executing ML Results query: {base_sql} with params: {query_params}")

    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True) # 返回字典形式的结果
            cursor.execute(base_sql, query_params)
            results = cursor.fetchall()
            
            # result_data_json 本身就是JSON字符串或已被connector处理为Python dict/list
            # 如果是字符串，AI可能需要json.loads()；如果是dict/list，可以直接用
            # 为确保一致性，我们尝试将数据库返回的JSON字符串（如果它是字符串的话）解析为Python对象
            for row in results:
                if isinstance(row.get("result_data_json"), str):
                    try:
                        row["result_data_json"] = json.loads(row["result_data_json"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse result_data_json for ml_result_id: {row.get('id')}")
                        # 保留原始字符串或设为None/错误标记
                if isinstance(row.get("result_timestamp_utc"), datetime.datetime):
                    row["result_timestamp_utc"] = row["result_timestamp_utc"].isoformat()
                if isinstance(row.get("created_at"), datetime.datetime):
                    row["created_at"] = row["created_at"].isoformat()

            logger.info(f"ML Results query returned {len(results)} records.")
            return {"results": results, "query_executed": base_sql.replace("%(limit)s", str(limit)).replace("%(offset)s", str(offset))}
        else:
            return {"error": "Failed to connect to the database."}
            
    except MySQLError as e:
        logger.error(f"Error querying ML results: {e}", exc_info=True)
        return {"error": f"Database query for ML results failed: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error during ML results query: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def save_ml_result_to_db(
    # 参数与 SaveMLResultInput 中的字段名完全对应
    execution_id: str,
    result_data: Dict[str, Any],
    result_type: Optional[str] = None,
    result_name: Optional[str] = None,
    ai_analysis_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    将AI分析和提取的机器学习结果以JSON格式保存到 ml_results 表。

    返回:
        一个字典，包含 "success" (布尔值), "ml_result_id" (新记录的ID) 或 "error" 及 "message"。
    """
    conn = None
    cursor = None
    ml_result_id = None
    
    # 0. (可选但推荐) 验证 execution_id 是否在 execution_runs 表中存在
    #    这确保了外键约束的有效性，避免了因不存在的execution_id导致插入失败。
    #    如果执行此验证，需要一个新的查询函数或在此处实现。
    #    为简化，此处暂时跳过，依赖数据库的外键约束（如果设置了ON DELETE/UPDATE规则，可能仍会插入成功但后续行为不确定）。
    #    更好的做法是在尝试插入前就进行验证。

    current_utc_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. 将 result_data 字典序列化为JSON字符串
    try:
        # ensure_ascii=False 允许正确存储非ASCII字符（例如中文）
        # indent=None （或不指定）可以使其更紧凑，如果不需要人类可读的格式化存储
        result_data_json_str = json.dumps(result_data, ensure_ascii=False)
    except TypeError as e:
        error_msg = f"传递给 'result_data' 的内容无法序列化为JSON: {e}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": "InvalidInputFormat", "message": error_msg}
    except Exception as e: # 其他可能的序列化错误
        error_msg = f"序列化 'result_data' 为JSON时发生未知错误: {e}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": "SerializationError", "message": error_msg}


    # 2. 准备SQL语句和参数
    sql = """
    INSERT INTO ml_results (
        execution_run_id, result_timestamp_utc, result_type, 
        result_name, result_data_json, ai_analysis_notes
    ) VALUES (
        %(execution_id)s, %(result_timestamp_utc)s, %(result_type)s,
        %(result_name)s, %(result_data_json)s, %(ai_analysis_notes)s
    )
    """
    params_to_insert = {
        "execution_id": execution_id,
        "result_timestamp_utc": current_utc_time_iso, # 使用ISO格式的字符串
        "result_type": result_type,
        "result_name": result_name,
        "result_data_json": result_data_json_str, # 传递JSON字符串
        "ai_analysis_notes": ai_analysis_notes
    }

    # 3. 执行数据库插入操作
    try:
        conn = get_db_connection()
        if not conn:
            # get_db_connection 内部已经log了错误
            return {"success": False, "error": "DatabaseConnectionError", "message": "无法连接到数据库。"}
            
        cursor = conn.cursor()
        cursor.execute(sql, params_to_insert)
        conn.commit()
        ml_result_id = cursor.lastrowid # 获取新插入行的ID
        
        success_msg = f"机器学习结果已成功保存到数据库。关联的执行ID: '{execution_id}', 新ML结果记录ID: {ml_result_id}."
        logger.info(success_msg)
        return {"success": True, "ml_result_id": ml_result_id, "message": success_msg}
            
    except MySQLError as e:
        error_code_str = f"DatabaseErrorCode: {e.errno}" if hasattr(e, 'errno') else "DatabaseError"
        error_msg = f"保存机器学习结果到MySQL时发生错误 (关联执行ID: '{execution_id}'): {e}"
        logger.error(error_msg, exc_info=True)
        if conn:
            try:
                conn.rollback() # 确保在发生错误时回滚
            except MySQLError as rb_e:
                logger.error(f"数据库回滚失败: {rb_e}", exc_info=True)
        return {"success": False, "error": error_code_str, "message": error_msg}
    except Exception as e: # 捕获其他所有可能的意外错误
        error_msg = f"保存机器学习结果时发生未知错误 (关联执行ID: '{execution_id}'): {e}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": "UnknownError", "message": error_msg}
    finally:
        if cursor:
            try: cursor.close()
            except: pass # 忽略关闭游标的错误
        if conn and conn.is_connected():
            try: conn.close()
            except: pass # 忽略关闭连接的错误