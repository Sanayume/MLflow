"""
Resource Monitoring and Management System
资源监控和管理系统
"""

import psutil
import docker
import threading
import time
import logging
import json
import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ResourceUsage:
    """资源使用情况"""
    timestamp: datetime.datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_io_bytes: Dict[str, int]
    gpu_usage: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class ResourceLimit:
    """资源限制"""
    resource_type: ResourceType
    limit_value: float
    alert_threshold: float
    critical_threshold: float
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['resource_type'] = self.resource_type.value
        return result

@dataclass
class ResourceAlert:
    """资源告警"""
    alert_id: str
    timestamp: datetime.datetime
    resource_type: ResourceType
    alert_level: AlertLevel
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime.datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['resource_type'] = self.resource_type.value
        result['alert_level'] = self.alert_level.value
        result['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return result

class ContainerResourceManager:
    """容器资源管理器"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            self.container_limits = {}
            logger.info("Container Resource Manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            self.docker_client = None
    
    def create_container_with_limits(self, image: str, command: str = None, 
                                   cpu_limit: float = None, memory_limit: str = None,
                                   **kwargs) -> Optional[str]:
        """创建带资源限制的容器"""
        if not self.docker_client:
            logger.error("Docker client not available")
            return None
        
        try:
            # 构建资源限制参数
            host_config_params = {}
            
            if cpu_limit:
                # CPU限制 (CPU份额，1.0 = 1个CPU核心)
                host_config_params['nano_cpus'] = int(cpu_limit * 1e9)
                
            if memory_limit:
                # 内存限制 (如 "512m", "2g")
                host_config_params['mem_limit'] = memory_limit
                
            # 创建host配置
            host_config = self.docker_client.api.create_host_config(**host_config_params)
            
            # 创建容器
            container = self.docker_client.containers.create(
                image=image,
                command=command,
                host_config=host_config,
                **kwargs
            )
            
            # 记录限制信息
            self.container_limits[container.id] = {
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit,
                'created_at': datetime.datetime.now()
            }
            
            logger.info(f"Created container {container.short_id} with limits: CPU={cpu_limit}, Memory={memory_limit}")
            return container.id
            
        except Exception as e:
            logger.error(f"Error creating container with limits: {str(e)}", exc_info=True)
            return None
    
    def get_container_stats(self, container_id: str) -> Optional[Dict[str, Any]]:
        """获取容器资源统计"""
        if not self.docker_client:
            return None
            
        try:
            container = self.docker_client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # 解析CPU使用率
            cpu_percent = self._calculate_cpu_percent(stats)
            
            # 解析内存使用情况
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_limit = stats['memory_stats'].get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            # 解析网络IO
            networks = stats.get('networks', {})
            network_io = {}
            for interface, data in networks.items():
                network_io[interface] = {
                    'rx_bytes': data.get('rx_bytes', 0),
                    'tx_bytes': data.get('tx_bytes', 0)
                }
            
            # 解析磁盘IO
            blk_io = stats.get('blkio_stats', {}).get('io_service_bytes_recursive', [])
            disk_read = sum(item['value'] for item in blk_io if item['op'] == 'Read')
            disk_write = sum(item['value'] for item in blk_io if item['op'] == 'Write')
            
            return {
                'container_id': container_id,
                'cpu_percent': cpu_percent,
                'memory_usage_bytes': memory_usage,
                'memory_limit_bytes': memory_limit,
                'memory_percent': memory_percent,
                'network_io': network_io,
                'disk_io': {
                    'read_bytes': disk_read,
                    'write_bytes': disk_write
                },
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting container stats: {str(e)}", exc_info=True)
            return None
    
    def _calculate_cpu_percent(self, stats: Dict[str, Any]) -> float:
        """计算CPU使用百分比"""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100
                return round(cpu_percent, 2)
            
            return 0.0
            
        except (KeyError, ZeroDivisionError):
            return 0.0
    
    def enforce_container_limits(self, container_id: str) -> bool:
        """强制执行容器限制"""
        if not self.docker_client or container_id not in self.container_limits:
            return False
            
        try:
            container = self.docker_client.containers.get(container_id)
            stats = self.get_container_stats(container_id)
            
            if not stats:
                return False
            
            limits = self.container_limits[container_id]
            actions_taken = []
            
            # 检查内存使用
            if stats['memory_percent'] > 95:  # 内存使用超过95%
                logger.warning(f"Container {container_id} memory usage critical: {stats['memory_percent']:.1f}%")
                # 可以在这里实施限制措施
                actions_taken.append("memory_warning_issued")
            
            # 检查CPU使用
            if stats['cpu_percent'] > 90:  # CPU使用超过90%
                logger.warning(f"Container {container_id} CPU usage high: {stats['cpu_percent']:.1f}%")
                actions_taken.append("cpu_warning_issued")
            
            return len(actions_taken) > 0
            
        except Exception as e:
            logger.error(f"Error enforcing container limits: {str(e)}", exc_info=True)
            return False

class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self, monitor_interval: int = 5, db_path: str = "agent_workspace/resource_monitor.db"):
        self.monitor_interval = monitor_interval
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.limits: Dict[ResourceType, ResourceLimit] = {}
        self.alerts: List[ResourceAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        self.monitoring = False
        self.monitor_thread = None
        
        self._init_database()
        self._set_default_limits()
        
        logger.info("System Resource Monitor initialized")
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 资源使用历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resource_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_gb REAL NOT NULL,
                    memory_total_gb REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    disk_used_gb REAL NOT NULL,
                    disk_total_gb REAL NOT NULL,
                    network_io_json TEXT,
                    gpu_usage_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 告警历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resource_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON resource_usage(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON resource_alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resource_type ON resource_alerts(resource_type)')
            
            conn.commit()
    
    def _set_default_limits(self):
        """设置默认资源限制"""
        self.limits = {
            ResourceType.CPU: ResourceLimit(
                resource_type=ResourceType.CPU,
                limit_value=90.0,
                alert_threshold=75.0,
                critical_threshold=85.0
            ),
            ResourceType.MEMORY: ResourceLimit(
                resource_type=ResourceType.MEMORY,
                limit_value=90.0,
                alert_threshold=75.0,
                critical_threshold=85.0
            ),
            ResourceType.DISK: ResourceLimit(
                resource_type=ResourceType.DISK,
                limit_value=85.0,
                alert_threshold=70.0,
                critical_threshold=80.0
            )
        }
    
    def set_resource_limit(self, resource_type: ResourceType, limit: ResourceLimit):
        """设置资源限制"""
        self.limits[resource_type] = limit
        logger.info(f"Set {resource_type.value} limit: {limit.limit_value}%")
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                usage = self._collect_resource_usage()
                
                if usage:
                    # 保存到数据库
                    self._save_usage_to_db(usage)
                    
                    # 检查告警
                    self._check_alerts(usage)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)
                time.sleep(self.monitor_interval)
    
    def _collect_resource_usage(self) -> Optional[ResourceUsage]:
        """收集资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # 网络IO
            net_io = psutil.net_io_counters()
            network_io_bytes = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # GPU使用情况（如果可用）
            gpu_usage = self._get_gpu_usage()
            
            return ResourceUsage(
                timestamp=datetime.datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_io_bytes=network_io_bytes,
                gpu_usage=gpu_usage
            )
            
        except Exception as e:
            logger.error(f"Error collecting resource usage: {str(e)}", exc_info=True)
            return None
    
    def _get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """获取GPU使用情况"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            
            device_count = nvml.nvmlDeviceGetCount()
            gpu_info = []
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取GPU信息
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # 获取内存信息
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used / (1024**3)  # GB
                memory_total = mem_info.total / (1024**3)  # GB
                memory_percent = (mem_info.used / mem_info.total) * 100
                
                # 获取GPU利用率
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # 获取温度
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                gpu_info.append({
                    'index': i,
                    'name': name,
                    'gpu_utilization': gpu_util,
                    'memory_used_gb': memory_used,
                    'memory_total_gb': memory_total,
                    'memory_percent': memory_percent,
                    'temperature_c': temp
                })
            
            return {'gpus': gpu_info, 'count': device_count}
            
        except ImportError:
            # NVIDIA ML library not available
            return None
        except Exception as e:
            logger.debug(f"Could not get GPU usage: {str(e)}")
            return None
    
    def _save_usage_to_db(self, usage: ResourceUsage):
        """保存使用情况到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO resource_usage (
                        timestamp, cpu_percent, memory_percent, memory_used_gb, memory_total_gb,
                        disk_percent, disk_used_gb, disk_total_gb, network_io_json, gpu_usage_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    usage.timestamp.isoformat(),
                    usage.cpu_percent,
                    usage.memory_percent,
                    usage.memory_used_gb,
                    usage.memory_total_gb,
                    usage.disk_percent,
                    usage.disk_used_gb,
                    usage.disk_total_gb,
                    json.dumps(usage.network_io_bytes),
                    json.dumps(usage.gpu_usage) if usage.gpu_usage else None
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving usage to database: {str(e)}", exc_info=True)
    
    def _check_alerts(self, usage: ResourceUsage):
        """检查告警条件"""
        current_values = {
            ResourceType.CPU: usage.cpu_percent,
            ResourceType.MEMORY: usage.memory_percent,
            ResourceType.DISK: usage.disk_percent
        }
        
        for resource_type, current_value in current_values.items():
            if resource_type not in self.limits or not self.limits[resource_type].enabled:
                continue
            
            limit = self.limits[resource_type]
            
            # 确定告警级别
            alert_level = None
            threshold_value = None
            
            if current_value >= limit.critical_threshold:
                alert_level = AlertLevel.CRITICAL
                threshold_value = limit.critical_threshold
            elif current_value >= limit.alert_threshold:
                alert_level = AlertLevel.WARNING
                threshold_value = limit.alert_threshold
            
            if alert_level:
                self._create_alert(resource_type, alert_level, current_value, threshold_value)
    
    def _create_alert(self, resource_type: ResourceType, alert_level: AlertLevel, 
                     current_value: float, threshold_value: float):
        """创建告警"""
        alert_id = f"{resource_type.value}_{alert_level.value}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 检查是否已有相同类型的未解决告警
        existing_alerts = [a for a in self.alerts 
                          if a.resource_type == resource_type and not a.resolved]
        
        if existing_alerts:
            # 更新现有告警
            latest_alert = existing_alerts[-1]
            if alert_level.value != latest_alert.alert_level.value:
                # 告警级别变化，创建新告警
                pass
            else:
                # 相同级别，不创建新告警
                return
        
        message = f"{resource_type.value.upper()} usage {alert_level.value}: {current_value:.1f}% (threshold: {threshold_value:.1f}%)"
        
        alert = ResourceAlert(
            alert_id=alert_id,
            timestamp=datetime.datetime.now(),
            resource_type=resource_type,
            alert_level=alert_level,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message
        )
        
        self.alerts.append(alert)
        self._save_alert_to_db(alert)
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}", exc_info=True)
        
        logger.warning(f"Resource alert created: {message}")
    
    def _save_alert_to_db(self, alert: ResourceAlert):
        """保存告警到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO resource_alerts (
                        alert_id, timestamp, resource_type, alert_level,
                        current_value, threshold_value, message, resolved, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.timestamp.isoformat(),
                    alert.resource_type.value,
                    alert.alert_level.value,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving alert to database: {str(e)}", exc_info=True)
    
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """获取当前资源使用情况"""
        return self._collect_resource_usage()
    
    def get_usage_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取使用历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
                
                cursor.execute('''
                    SELECT * FROM resource_usage 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                ''', (start_time.isoformat(),))
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting usage history: {str(e)}", exc_info=True)
            return []
    
    def get_alerts(self, resolved: bool = None, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                start_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
                where_clauses = ["timestamp >= ?"]
                params = [start_time.isoformat()]
                
                if resolved is not None:
                    where_clauses.append("resolved = ?")
                    params.append(resolved)
                
                query = f'''
                    SELECT * FROM resource_alerts 
                    WHERE {" AND ".join(where_clauses)}
                    ORDER BY timestamp DESC
                '''
                
                cursor.execute(query, params)
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}", exc_info=True)
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        try:
            # 更新内存中的告警
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.datetime.now()
                    break
            
            # 更新数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE resource_alerts 
                    SET resolved = 1, resolved_at = ? 
                    WHERE alert_id = ?
                ''', (datetime.datetime.now().isoformat(), alert_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Alert {alert_id} resolved")
                    return True
                
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}", exc_info=True)
        
        return False
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
                cutoff_str = cutoff_date.isoformat()
                
                # 清理旧的使用记录
                cursor.execute('DELETE FROM resource_usage WHERE timestamp < ?', (cutoff_str,))
                usage_deleted = cursor.rowcount
                
                # 清理旧的已解决告警
                cursor.execute('DELETE FROM resource_alerts WHERE timestamp < ? AND resolved = 1', (cutoff_str,))
                alerts_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up {usage_deleted} usage records and {alerts_deleted} resolved alerts")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}", exc_info=True)

# 使用示例
if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(signum, frame):
        print("\nShutting down monitor...")
        monitor.stop_monitoring()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建监控器
    monitor = SystemResourceMonitor(monitor_interval=5)
    
    # 添加告警回调
    def alert_callback(alert: ResourceAlert):
        print(f"🚨 ALERT: {alert.message}")
    
    monitor.add_alert_callback(alert_callback)
    
    # 开始监控
    monitor.start_monitoring()
    
    print("Resource monitoring started. Press Ctrl+C to stop.")
    print("Current resource usage:")
    
    try:
        while True:
            usage = monitor.get_current_usage()
            if usage:
                print(f"\rCPU: {usage.cpu_percent:5.1f}% | "
                      f"Memory: {usage.memory_percent:5.1f}% | "
                      f"Disk: {usage.disk_percent:5.1f}%", end="")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()
        print("\nMonitoring stopped.")