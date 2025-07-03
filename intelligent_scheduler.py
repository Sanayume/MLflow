# intelligent_scheduler.py
"""
智能ML工具调度系统
基于gemini-cli的TypeScript工具调度架构
提供智能依赖分析、资源管理和执行优化
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Callable, Union
import networkx as nx
from datetime import datetime, timedelta
import json
import uuid
import threading
import queue
import psutil
from pathlib import Path

class ToolStatus(Enum):
    """工具执行状态"""
    PENDING = "pending"
    VALIDATING = "validating"
    SCHEDULED = "scheduled"
    WAITING_APPROVAL = "waiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class ResourceRequirement:
    """资源需求定义"""
    type: ResourceType
    amount: float
    unit: str  # cores, GB, %
    exclusive: bool = False  # 是否独占
    
@dataclass
class MLToolCall:
    """ML工具调用定义"""
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    
    # 依赖关系
    explicit_dependencies: Set[str] = field(default_factory=set)
    implicit_dependencies: Set[str] = field(default_factory=set)
    
    # 执行属性
    priority: int = 0  # 优先级 (越高越优先)
    estimated_duration: float = 60.0  # 预估执行时间(秒)
    timeout: float = 3600.0  # 超时时间(秒)
    
    # 资源需求
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # 执行信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_duration: Optional[float] = None
    
    # 结果和错误
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # 批准相关
    requires_approval: bool = False
    approval_message: Optional[str] = None
    approved: bool = False
    
    # 回调函数
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None

@dataclass
class ResourcePool:
    """资源池管理"""
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    disk_gb: float
    
    # 当前使用情况
    cpu_used: float = 0.0
    memory_used: float = 0.0
    gpu_used: int = 0
    disk_used: float = 0.0
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """获取可用资源"""
        return {
            ResourceType.CPU: self.cpu_cores - self.cpu_used,
            ResourceType.MEMORY: self.memory_gb - self.memory_used,
            ResourceType.GPU: self.gpu_count - self.gpu_used,
            ResourceType.DISK: self.disk_gb - self.disk_used
        }
    
    def can_allocate(self, requirements: List[ResourceRequirement]) -> bool:
        """检查是否可以分配资源"""
        available = self.get_available_resources()
        
        for req in requirements:
            if available[req.type] < req.amount:
                return False
        return True
    
    def allocate(self, requirements: List[ResourceRequirement]) -> bool:
        """分配资源"""
        if not self.can_allocate(requirements):
            return False
        
        for req in requirements:
            if req.type == ResourceType.CPU:
                self.cpu_used += req.amount
            elif req.type == ResourceType.MEMORY:
                self.memory_used += req.amount
            elif req.type == ResourceType.GPU:
                self.gpu_used += int(req.amount)
            elif req.type == ResourceType.DISK:
                self.disk_used += req.amount
        
        return True
    
    def release(self, requirements: List[ResourceRequirement]):
        """释放资源"""
        for req in requirements:
            if req.type == ResourceType.CPU:
                self.cpu_used = max(0, self.cpu_used - req.amount)
            elif req.type == ResourceType.MEMORY:
                self.memory_used = max(0, self.memory_used - req.amount)
            elif req.type == ResourceType.GPU:
                self.gpu_used = max(0, self.gpu_used - int(req.amount))
            elif req.type == ResourceType.DISK:
                self.disk_used = max(0, self.disk_used - req.amount)

class IntelligentMLToolScheduler:
    """智能ML工具调度器"""
    
    def __init__(self, max_concurrent_tools: int = 4):
        self.max_concurrent_tools = max_concurrent_tools
        
        # 工具管理
        self.tools: Dict[str, MLToolCall] = {}
        self.dependency_graph = nx.DiGraph()
        self.execution_queue = queue.PriorityQueue()
        
        # 资源管理
        self.resource_pool = self._initialize_resource_pool()
        
        # 执行管理
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.completed_tools: List[str] = []
        self.failed_tools: List[str] = []
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            'tool_scheduled': [],
            'tool_started': [],
            'tool_completed': [],
            'tool_failed': [],
            'batch_completed': []
        }
        
        # 统计信息
        self.stats = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'avg_execution_time': 0.0,
            'resource_utilization': {}
        }
        
    def _initialize_resource_pool(self) -> ResourcePool:
        """初始化资源池"""
        return ResourcePool(
            cpu_cores=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_count=0,  # 需要GPU检测
            disk_gb=psutil.disk_usage('/').free / (1024**3)
        )
    
    def register_event_callback(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self.event_callbacks:
            self.event_callbacks[event].append(callback)
    
    async def schedule_tools(self, tool_calls: List[MLToolCall]) -> List[str]:
        """智能调度ML工具执行"""
        print(f"🎯 开始智能调度 {len(tool_calls)} 个工具")
        
        # 1. 注册所有工具
        for tool in tool_calls:
            self.tools[tool.id] = tool
            self.stats['total_scheduled'] += 1
        
        # 2. 构建依赖图
        await self._build_dependency_graph(tool_calls)
        
        # 3. 检测循环依赖
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            cycles = list(nx.simple_cycles(self.dependency_graph))
            raise ValueError(f"检测到循环依赖: {cycles}")
        
        # 4. 拓扑排序获得基础执行顺序
        base_order = list(nx.topological_sort(self.dependency_graph))
        
        # 5. 智能优化执行顺序
        optimized_order = await self._optimize_execution_order(base_order)
        
        # 6. 创建并行执行组
        parallel_groups = await self._create_parallel_groups(optimized_order)
        
        # 7. 执行调度
        scheduled_ids = await self._execute_parallel_groups(parallel_groups)
        
        print(f"✅ 调度完成，成功调度 {len(scheduled_ids)} 个工具")
        return scheduled_ids
    
    async def _build_dependency_graph(self, tool_calls: List[MLToolCall]):
        """构建智能依赖图"""
        print("🔗 构建工具依赖图...")
        
        # 清空现有图
        self.dependency_graph.clear()
        
        # 添加所有节点
        for tool in tool_calls:
            self.dependency_graph.add_node(tool.id, tool=tool)
        
        # 添加显式依赖
        for tool in tool_calls:
            for dep_id in tool.explicit_dependencies:
                if dep_id in self.tools:
                    self.dependency_graph.add_edge(dep_id, tool.id)
        
        # 推断隐式依赖
        for tool in tool_calls:
            implicit_deps = await self._infer_implicit_dependencies(tool, tool_calls)
            for dep_id in implicit_deps:
                self.dependency_graph.add_edge(dep_id, tool.id)
                tool.implicit_dependencies.add(dep_id)
        
        print(f"📊 依赖图构建完成: {len(self.dependency_graph.nodes)} 节点, {len(self.dependency_graph.edges)} 边")
    
    async def _infer_implicit_dependencies(self, tool: MLToolCall, all_tools: List[MLToolCall]) -> Set[str]:
        """智能推断工具间的隐式依赖"""
        implicit_deps = set()
        
        # 数据流依赖分析
        data_flow_deps = self._analyze_data_flow_dependencies(tool, all_tools)
        implicit_deps.update(data_flow_deps)
        
        # 资源冲突分析
        resource_conflict_deps = self._analyze_resource_conflicts(tool, all_tools)
        implicit_deps.update(resource_conflict_deps)
        
        # 文件系统依赖
        file_system_deps = self._analyze_file_system_dependencies(tool, all_tools)
        implicit_deps.update(file_system_deps)
        
        # 模型生命周期依赖
        model_lifecycle_deps = self._analyze_model_lifecycle_dependencies(tool, all_tools)
        implicit_deps.update(model_lifecycle_deps)
        
        return implicit_deps
    
    def _analyze_data_flow_dependencies(self, tool: MLToolCall, all_tools: List[MLToolCall]) -> Set[str]:
        """分析数据流依赖"""
        deps = set()
        
        # 常见的数据流模式
        data_flow_patterns = {
            'train_model': ['preprocess_data', 'feature_engineering', 'data_validation'],
            'evaluate_model': ['train_model'],
            'deploy_model': ['train_model', 'evaluate_model'],
            'generate_report': ['evaluate_model', 'analyze_results'],
            'visualize_results': ['evaluate_model', 'analyze_data'],
            'feature_engineering': ['preprocess_data', 'load_data'],
            'hyperparameter_tuning': ['preprocess_data', 'feature_engineering']
        }
        
        if tool.tool_name in data_flow_patterns:
            required_predecessors = data_flow_patterns[tool.tool_name]
            for other in all_tools:
                if other.tool_name in required_predecessors:
                    # 检查是否操作相同的数据集
                    if self._shares_data_pipeline(tool, other):
                        deps.add(other.id)
        
        return deps
    
    def _analyze_resource_conflicts(self, tool: MLToolCall, all_tools: List[MLToolCall]) -> Set[str]:
        """分析资源冲突"""
        deps = set()
        
        for other in all_tools:
            if other.id == tool.id:
                continue
                
            # 检查资源冲突
            if self._has_resource_conflict(tool, other):
                # 按优先级确定依赖关系
                if other.priority > tool.priority:
                    deps.add(other.id)
                elif other.priority == tool.priority:
                    # 优先级相同时，按预估执行时间排序
                    if other.estimated_duration > tool.estimated_duration:
                        deps.add(other.id)
        
        return deps
    
    def _analyze_file_system_dependencies(self, tool: MLToolCall, all_tools: List[MLToolCall]) -> Set[str]:
        """分析文件系统依赖"""
        deps = set()
        
        tool_input_files = self._extract_input_files(tool)
        
        for other in all_tools:
            if other.id == tool.id:
                continue
                
            other_output_files = self._extract_output_files(other)
            
            # 如果当前工具的输入依赖其他工具的输出
            if tool_input_files & other_output_files:
                deps.add(other.id)
        
        return deps
    
    def _analyze_model_lifecycle_dependencies(self, tool: MLToolCall, all_tools: List[MLToolCall]) -> Set[str]:
        """分析模型生命周期依赖"""
        deps = set()
        
        model_lifecycle_order = [
            'data_validation', 'preprocess_data', 'feature_engineering',
            'train_model', 'validate_model', 'evaluate_model', 
            'hyperparameter_tuning', 'deploy_model', 'monitor_model'
        ]
        
        try:
            tool_stage = model_lifecycle_order.index(tool.tool_name)
            
            for other in all_tools:
                if other.id == tool.id:
                    continue
                    
                try:
                    other_stage = model_lifecycle_order.index(other.tool_name)
                    # 如果其他工具在生命周期中更早，且操作相同模型
                    if other_stage < tool_stage and self._shares_model_artifacts(tool, other):
                        deps.add(other.id)
                except ValueError:
                    # 其他工具不在标准生命周期中
                    pass
        except ValueError:
            # 当前工具不在标准生命周期中
            pass
        
        return deps
    
    def _shares_data_pipeline(self, tool1: MLToolCall, tool2: MLToolCall) -> bool:
        """检查两个工具是否共享数据管道"""
        # 检查参数中的数据集路径
        dataset_keys = ['dataset_path', 'data_path', 'input_data', 'data_file']
        
        tool1_datasets = set()
        tool2_datasets = set()
        
        for key in dataset_keys:
            if key in tool1.parameters:
                tool1_datasets.add(str(tool1.parameters[key]))
            if key in tool2.parameters:
                tool2_datasets.add(str(tool2.parameters[key]))
        
        return bool(tool1_datasets & tool2_datasets)
    
    def _has_resource_conflict(self, tool1: MLToolCall, tool2: MLToolCall) -> bool:
        """检查两个工具是否有资源冲突"""
        for req1 in tool1.resource_requirements:
            for req2 in tool2.resource_requirements:
                if req1.type == req2.type:
                    # 检查是否有独占需求或资源使用过多
                    if req1.exclusive or req2.exclusive:
                        return True
                    
                    # 检查总资源需求是否超过可用资源
                    available = self.resource_pool.get_available_resources()[req1.type]
                    if (req1.amount + req2.amount) > available:
                        return True
        
        return False
    
    def _extract_input_files(self, tool: MLToolCall) -> Set[str]:
        """提取工具的输入文件"""
        input_files = set()
        input_keys = ['input_file', 'data_file', 'model_file', 'config_file']
        
        for key in input_keys:
            if key in tool.parameters:
                input_files.add(str(tool.parameters[key]))
        
        return input_files
    
    def _extract_output_files(self, tool: MLToolCall) -> Set[str]:
        """提取工具的输出文件"""
        output_files = set()
        output_keys = ['output_file', 'model_path', 'result_path', 'report_path']
        
        for key in output_keys:
            if key in tool.parameters:
                output_files.add(str(tool.parameters[key]))
        
        return output_files
    
    def _shares_model_artifacts(self, tool1: MLToolCall, tool2: MLToolCall) -> bool:
        """检查两个工具是否共享模型文件"""
        model_keys = ['model_path', 'model_file', 'checkpoint_path']
        
        tool1_models = set()
        tool2_models = set()
        
        for key in model_keys:
            if key in tool1.parameters:
                tool1_models.add(str(tool1.parameters[key]))
            if key in tool2.parameters:
                tool2_models.add(str(tool2.parameters[key]))
        
        return bool(tool1_models & tool2_models)
    
    async def _optimize_execution_order(self, base_order: List[str]) -> List[str]:
        """智能优化执行顺序"""
        print("🧠 优化执行顺序...")
        
        # 创建优化后的顺序
        optimized = []
        remaining = set(base_order)
        
        while remaining:
            # 找到可以执行的工具（所有依赖都已完成）
            ready_tools = []
            for tool_id in remaining:
                tool = self.tools[tool_id]
                all_deps = tool.explicit_dependencies | tool.implicit_dependencies
                
                # 检查所有依赖是否已经在优化序列中
                if all_deps.issubset(set(optimized)):
                    ready_tools.append(tool_id)
            
            if not ready_tools:
                # 如果没有准备好的工具，可能有循环依赖
                break
            
            # 按优先级和资源需求排序
            ready_tools.sort(key=lambda tid: (
                -self.tools[tid].priority,  # 优先级高的在前
                self.tools[tid].estimated_duration,  # 执行时间短的在前
                -len(self.tools[tid].resource_requirements)  # 资源需求少的在前
            ))
            
            # 选择最优的工具
            selected = ready_tools[0]
            optimized.append(selected)
            remaining.remove(selected)
        
        # 将剩余的工具按原顺序添加（处理循环依赖情况）
        for tool_id in base_order:
            if tool_id in remaining:
                optimized.append(tool_id)
        
        print(f"✅ 执行顺序优化完成")
        return optimized
    
    async def _create_parallel_groups(self, execution_order: List[str]) -> List[List[str]]:
        """创建并行执行组"""
        print("🔄 创建并行执行组...")
        
        groups = []
        remaining = execution_order.copy()
        completed = set()
        
        while remaining:
            current_group = []
            current_resources = ResourcePool(
                cpu_cores=self.resource_pool.cpu_cores,
                memory_gb=self.resource_pool.memory_gb,
                gpu_count=self.resource_pool.gpu_count,
                disk_gb=self.resource_pool.disk_gb
            )
            
            # 找到可以并行执行的工具
            i = 0
            while i < len(remaining) and len(current_group) < self.max_concurrent_tools:
                tool_id = remaining[i]
                tool = self.tools[tool_id]
                
                # 检查依赖是否满足
                all_deps = tool.explicit_dependencies | tool.implicit_dependencies
                if not all_deps.issubset(completed):
                    i += 1
                    continue
                
                # 检查资源是否可用
                if current_resources.can_allocate(tool.resource_requirements):
                    current_group.append(tool_id)
                    current_resources.allocate(tool.resource_requirements)
                    remaining.pop(i)
                else:
                    i += 1
            
            if current_group:
                groups.append(current_group)
                completed.update(current_group)
            else:
                # 如果没有工具可以执行，强制选择第一个
                if remaining:
                    groups.append([remaining.pop(0)])
                    completed.add(groups[-1][0])
        
        print(f"📊 创建了 {len(groups)} 个并行执行组")
        return groups
    
    async def _execute_parallel_groups(self, groups: List[List[str]]) -> List[str]:
        """执行并行组"""
        scheduled_ids = []
        
        for i, group in enumerate(groups):
            print(f"🚀 执行第 {i+1}/{len(groups)} 组: {len(group)} 个工具")
            
            # 检查批准需求
            approval_needed = []
            for tool_id in group:
                tool = self.tools[tool_id]
                if tool.requires_approval and not tool.approved:
                    approval_needed.append(tool_id)
            
            # 处理批准
            if approval_needed:
                await self._request_approvals(approval_needed)
            
            # 并行执行组内工具
            group_results = await self._execute_group(group)
            scheduled_ids.extend(group_results)
        
        return scheduled_ids
    
    async def _request_approvals(self, tool_ids: List[str]):
        """请求工具执行批准"""
        print(f"⚠️ 需要批准 {len(tool_ids)} 个工具的执行")
        
        for tool_id in tool_ids:
            tool = self.tools[tool_id]
            print(f"🔍 工具: {tool.tool_name}")
            print(f"📝 说明: {tool.approval_message or '需要用户确认'}")
            
            # 在实际应用中，这里应该是异步的用户界面交互
            # 现在暂时自动批准
            tool.approved = True
            print(f"✅ 工具 {tool.tool_name} 已批准")
    
    async def _execute_group(self, group: List[str]) -> List[str]:
        """执行工具组"""
        tasks = []
        
        # 分配资源
        for tool_id in group:
            tool = self.tools[tool_id]
            if not self.resource_pool.allocate(tool.resource_requirements):
                print(f"⚠️ 无法为工具 {tool.tool_name} 分配资源")
                continue
            
            # 创建执行任务
            task = asyncio.create_task(self._execute_single_tool(tool))
            tasks.append((tool_id, task))
            
            # 触发事件
            await self._trigger_event('tool_scheduled', tool)
        
        # 等待所有任务完成
        results = []
        for tool_id, task in tasks:
            try:
                result = await task
                results.append(tool_id)
                self.completed_tools.append(tool_id)
                self.stats['total_completed'] += 1
                
                # 触发完成事件
                await self._trigger_event('tool_completed', self.tools[tool_id])
                
            except Exception as e:
                print(f"❌ 工具 {tool_id} 执行失败: {str(e)}")
                self.tools[tool_id].status = ToolStatus.FAILED
                self.tools[tool_id].error = str(e)
                self.failed_tools.append(tool_id)
                self.stats['total_failed'] += 1
                
                # 触发失败事件
                await self._trigger_event('tool_failed', self.tools[tool_id])
            
            finally:
                # 释放资源
                tool = self.tools[tool_id]
                self.resource_pool.release(tool.resource_requirements)
        
        return results
    
    async def _execute_single_tool(self, tool: MLToolCall) -> Any:
        """执行单个工具"""
        tool.status = ToolStatus.EXECUTING
        tool.start_time = datetime.now()
        
        # 触发开始事件
        await self._trigger_event('tool_started', tool)
        
        try:
            # 模拟工具执行（实际应用中这里会调用真实的工具）
            if tool.progress_callback:
                # 模拟进度更新
                for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    await asyncio.sleep(tool.estimated_duration * 0.2)
                    await tool.progress_callback(tool.id, progress)
            else:
                await asyncio.sleep(tool.estimated_duration)
            
            # 模拟成功结果
            tool.result = {"status": "success", "data": f"Result from {tool.tool_name}"}
            tool.status = ToolStatus.COMPLETED
            
        except asyncio.TimeoutError:
            tool.status = ToolStatus.TIMEOUT
            tool.error = "执行超时"
            raise
        except Exception as e:
            tool.status = ToolStatus.FAILED
            tool.error = str(e)
            raise
        finally:
            tool.end_time = datetime.now()
            if tool.start_time:
                tool.actual_duration = (tool.end_time - tool.start_time).total_seconds()
            
            if tool.completion_callback:
                await tool.completion_callback(tool.id, tool.result)
        
        return tool.result
    
    async def _trigger_event(self, event: str, tool: MLToolCall):
        """触发事件回调"""
        if event in self.event_callbacks:
            for callback in self.event_callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(tool)
                    else:
                        callback(tool)
                except Exception as e:
                    print(f"⚠️ 事件回调失败 {event}: {str(e)}")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """获取执行状态"""
        status_counts = {}
        for status in ToolStatus:
            status_counts[status.value] = len([
                t for t in self.tools.values() if t.status == status
            ])
        
        return {
            'total_tools': len(self.tools),
            'status_breakdown': status_counts,
            'resource_utilization': {
                'cpu': f"{self.resource_pool.cpu_used}/{self.resource_pool.cpu_cores}",
                'memory': f"{self.resource_pool.memory_used:.1f}/{self.resource_pool.memory_gb:.1f} GB",
                'gpu': f"{self.resource_pool.gpu_used}/{self.resource_pool.gpu_count}"
            },
            'stats': self.stats
        }
    
    def export_execution_report(self, output_path: str = None) -> str:
        """导出执行报告"""
        if output_path is None:
            output_path = f"ml_execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_summary': self.get_execution_status(),
            'tool_details': {
                tool_id: {
                    'tool_name': tool.tool_name,
                    'status': tool.status.value,
                    'duration': tool.actual_duration,
                    'start_time': tool.start_time.isoformat() if tool.start_time else None,
                    'end_time': tool.end_time.isoformat() if tool.end_time else None,
                    'error': tool.error,
                    'dependencies': list(tool.explicit_dependencies | tool.implicit_dependencies)
                }
                for tool_id, tool in self.tools.items()
            },
            'dependency_graph': {
                'nodes': list(self.dependency_graph.nodes()),
                'edges': list(self.dependency_graph.edges())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 执行报告已导出到: {output_path}")
        return output_path

# 测试函数
async def test_intelligent_scheduler():
    """测试智能调度器"""
    print("🚀 开始智能调度器测试...")
    
    scheduler = IntelligentMLToolScheduler(max_concurrent_tools=3)
    
    # 创建测试工具
    tools = []
    
    # 数据预处理工具
    preprocess_tool = MLToolCall(
        id="preprocess_1",
        tool_name="preprocess_data",
        parameters={"dataset_path": "data/raw.csv", "output_path": "data/processed.csv"},
        estimated_duration=10,
        priority=3,
        resource_requirements=[
            ResourceRequirement(ResourceType.CPU, 2, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 4, "GB")
        ]
    )
    tools.append(preprocess_tool)
    
    # 特征工程工具
    feature_tool = MLToolCall(
        id="feature_1",
        tool_name="feature_engineering",
        parameters={"input_data": "data/processed.csv", "output_path": "data/features.csv"},
        estimated_duration=15,
        priority=2,
        explicit_dependencies={"preprocess_1"},
        resource_requirements=[
            ResourceRequirement(ResourceType.CPU, 1, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 2, "GB")
        ]
    )
    tools.append(feature_tool)
    
    # 模型训练工具
    train_tool = MLToolCall(
        id="train_1",
        tool_name="train_model",
        parameters={"data_path": "data/features.csv", "model_path": "models/model.pkl"},
        estimated_duration=30,
        priority=1,
        requires_approval=True,
        approval_message="训练模型会消耗大量计算资源",
        resource_requirements=[
            ResourceRequirement(ResourceType.CPU, 4, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 8, "GB")
        ]
    )
    tools.append(train_tool)
    
    # 模型评估工具
    eval_tool = MLToolCall(
        id="eval_1",
        tool_name="evaluate_model",
        parameters={"model_path": "models/model.pkl", "test_data": "data/test.csv"},
        estimated_duration=5,
        priority=2,
        resource_requirements=[
            ResourceRequirement(ResourceType.CPU, 1, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 2, "GB")
        ]
    )
    tools.append(eval_tool)
    
    # 注册事件回调
    async def on_tool_completed(tool: MLToolCall):
        print(f"✅ 工具完成: {tool.tool_name} (耗时: {tool.actual_duration:.1f}s)")
    
    scheduler.register_event_callback('tool_completed', on_tool_completed)
    
    # 执行调度
    try:
        scheduled_ids = await scheduler.schedule_tools(tools)
        print(f"🎉 调度完成: {len(scheduled_ids)} 个工具成功调度")
        
        # 打印执行状态
        status = scheduler.get_execution_status()
        print(f"📊 执行状态: {status}")
        
        # 导出报告
        report_path = scheduler.export_execution_report()
        print(f"📋 执行报告: {report_path}")
        
    except Exception as e:
        print(f"❌ 调度失败: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_scheduler())