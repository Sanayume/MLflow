"""
Enhanced AI Agent with improved reasoning, memory, and planning capabilities.
增强型AI代理，具备改进的推理、记忆和规划能力。
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMemory

# 导入智能提示系统
from intelligent_prompts import IntelligentPromptGenerator, PromptTemplate

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型枚举"""
    DATA_EXPLORATION = "data_exploration"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    VISUALIZATION = "visualization"
    DEPLOYMENT = "deployment"
    ANALYSIS = "analysis"
    OTHER = "other"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"  # 当前会话
    WORKING_MEMORY = "working_memory"  # 工作记忆
    LONG_TERM = "long_term"  # 长期记忆
    EPISODIC = "episodic"  # 情节记忆
    SEMANTIC = "semantic"  # 语义记忆

@dataclass
class TaskPlan:
    """任务规划数据结构"""
    task_id: str
    task_type: TaskType
    description: str
    status: TaskStatus
    priority: int  # 1-10, 10最高
    estimated_duration: Optional[int]  # 估计耗时（分钟）
    dependencies: List[str]  # 依赖的任务ID
    subtasks: List['TaskPlan']
    created_at: datetime.datetime
    updated_at: datetime.datetime
    completion_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['task_type'] = self.task_type.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        result['subtasks'] = [subtask.to_dict() for subtask in self.subtasks]
        return result

@dataclass
class MemoryEntry:
    """记忆条目数据结构"""
    entry_id: str
    memory_type: MemoryType
    content: str
    importance: float  # 0-1, 重要性评分
    timestamp: datetime.datetime
    tags: List[str]
    related_tasks: List[str]
    access_count: int = 0
    last_accessed: Optional[datetime.datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['memory_type'] = self.memory_type.value
        result['timestamp'] = self.timestamp.isoformat()
        result['last_accessed'] = self.last_accessed.isoformat() if self.last_accessed else None
        return result

class EnhancedMemorySystem:
    """增强的记忆系统"""
    
    def __init__(self, max_short_term: int = 50, max_working: int = 20):
        self.memories: Dict[str, MemoryEntry] = {}
        self.max_short_term = max_short_term
        self.max_working = max_working
        self.importance_threshold = 0.7  # 重要性阈值，超过此值的记忆会被保留更久
        
    def add_memory(self, content: str, memory_type: MemoryType, 
                   importance: float = 0.5, tags: List[str] = None,
                   related_tasks: List[str] = None) -> str:
        """添加记忆"""
        entry_id = f"mem_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        memory = MemoryEntry(
            entry_id=entry_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            timestamp=datetime.datetime.now(),
            tags=tags or [],
            related_tasks=related_tasks or []
        )
        
        self.memories[entry_id] = memory
        self._manage_memory_capacity()
        
        logger.info(f"Added memory: {entry_id} (type: {memory_type.value}, importance: {importance})")
        return entry_id
    
    def retrieve_memories(self, query: str = None, memory_type: MemoryType = None,
                         tags: List[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """检索记忆"""
        memories = list(self.memories.values())
        
        # 过滤条件
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        if query:
            # 简单的文本匹配，可以后续增强为语义匹配
            memories = [m for m in memories if query.lower() in m.content.lower()]
        
        # 按重要性和最近访问时间排序
        memories.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        
        # 更新访问记录
        for memory in memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.datetime.now()
        
        return memories[:limit]
    
    def _manage_memory_capacity(self):
        """管理记忆容量，删除不重要的旧记忆"""
        short_term_memories = [m for m in self.memories.values() 
                              if m.memory_type == MemoryType.SHORT_TERM]
        working_memories = [m for m in self.memories.values() 
                           if m.memory_type == MemoryType.WORKING_MEMORY]
        
        # 管理短期记忆
        if len(short_term_memories) > self.max_short_term:
            # 保留重要的记忆，删除不重要的旧记忆
            short_term_memories.sort(key=lambda x: (x.importance, x.timestamp))
            to_remove = short_term_memories[:len(short_term_memories) - self.max_short_term]
            for memory in to_remove:
                if memory.importance < self.importance_threshold:
                    del self.memories[memory.entry_id]
        
        # 管理工作记忆
        if len(working_memories) > self.max_working:
            working_memories.sort(key=lambda x: (x.importance, x.timestamp))
            to_remove = working_memories[:len(working_memories) - self.max_working]
            for memory in to_remove:
                if memory.importance < self.importance_threshold:
                    del self.memories[memory.entry_id]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        memory_types = {}
        for memory in self.memories.values():
            mem_type = memory.memory_type.value
            if mem_type not in memory_types:
                memory_types[mem_type] = 0
            memory_types[mem_type] += 1
        
        return {
            "total_memories": len(self.memories),
            "memory_types": memory_types,
            "high_importance_count": len([m for m in self.memories.values() 
                                        if m.importance >= self.importance_threshold])
        }

class EnhancedTaskPlanner:
    """增强的任务规划器"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskPlan] = {}
        self.task_templates = self._load_task_templates()
        
    def _load_task_templates(self) -> Dict[TaskType, Dict[str, Any]]:
        """加载任务模板"""
        return {
            TaskType.DATA_EXPLORATION: {
                "common_steps": [
                    "检查数据形状和基本信息",
                    "查看数据类型和缺失值",
                    "生成描述性统计",
                    "创建数据分布可视化"
                ],
                "estimated_duration": 15
            },
            TaskType.DATA_PREPROCESSING: {
                "common_steps": [
                    "处理缺失值",
                    "处理异常值",
                    "数据类型转换",
                    "数据归一化/标准化"
                ],
                "estimated_duration": 30
            },
            TaskType.FEATURE_ENGINEERING: {
                "common_steps": [
                    "特征选择",
                    "特征创建",
                    "特征转换",
                    "特征重要性分析"
                ],
                "estimated_duration": 45
            },
            TaskType.MODEL_TRAINING: {
                "common_steps": [
                    "选择合适的算法",
                    "划分训练测试集",
                    "训练模型",
                    "模型验证"
                ],
                "estimated_duration": 60
            }
        }
    
    def create_task_plan(self, description: str, task_type: TaskType = None,
                        priority: int = 5, dependencies: List[str] = None) -> TaskPlan:
        """创建任务计划"""
        task_id = f"task_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 如果没有指定任务类型，尝试从描述中推断
        if not task_type:
            task_type = self._infer_task_type(description)
        
        # 获取模板信息
        template = self.task_templates.get(task_type, {})
        estimated_duration = template.get("estimated_duration", 30)
        
        # 创建子任务
        subtasks = []
        if "common_steps" in template:
            for i, step in enumerate(template["common_steps"]):
                subtask_id = f"{task_id}_subtask_{i+1}"
                subtask = TaskPlan(
                    task_id=subtask_id,
                    task_type=task_type,
                    description=step,
                    status=TaskStatus.PENDING,
                    priority=priority,
                    estimated_duration=estimated_duration // len(template["common_steps"]),
                    dependencies=[],
                    subtasks=[],
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now()
                )
                subtasks.append(subtask)
        
        task_plan = TaskPlan(
            task_id=task_id,
            task_type=task_type,
            description=description,
            status=TaskStatus.PENDING,
            priority=priority,
            estimated_duration=estimated_duration,
            dependencies=dependencies or [],
            subtasks=subtasks,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        
        self.tasks[task_id] = task_plan
        logger.info(f"Created task plan: {task_id} ({task_type.value})")
        return task_plan
    
    def _infer_task_type(self, description: str) -> TaskType:
        """从描述中推断任务类型"""
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ["explore", "分析", "查看", "统计"]):
            return TaskType.DATA_EXPLORATION
        elif any(keyword in description_lower for keyword in ["clean", "preprocess", "清洗", "预处理"]):
            return TaskType.DATA_PREPROCESSING
        elif any(keyword in description_lower for keyword in ["feature", "特征", "变量"]):
            return TaskType.FEATURE_ENGINEERING
        elif any(keyword in description_lower for keyword in ["train", "model", "训练", "模型"]):
            return TaskType.MODEL_TRAINING
        elif any(keyword in description_lower for keyword in ["evaluate", "assess", "评估", "测试"]):
            return TaskType.MODEL_EVALUATION
        elif any(keyword in description_lower for keyword in ["visualize", "plot", "chart", "可视化", "图表"]):
            return TaskType.VISUALIZATION
        else:
            return TaskType.OTHER
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          completion_percentage: float = None):
        """更新任务状态"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.datetime.now()
            if completion_percentage is not None:
                self.tasks[task_id].completion_percentage = completion_percentage
            logger.info(f"Updated task {task_id} status to {status.value}")
    
    def get_next_tasks(self, limit: int = 5) -> List[TaskPlan]:
        """获取下一个待执行的任务"""
        pending_tasks = [task for task in self.tasks.values() 
                        if task.status == TaskStatus.PENDING]
        
        # 检查依赖关系
        available_tasks = []
        for task in pending_tasks:
            if self._are_dependencies_satisfied(task.dependencies):
                available_tasks.append(task)
        
        # 按优先级排序
        available_tasks.sort(key=lambda x: x.priority, reverse=True)
        return available_tasks[:limit]
    
    def _are_dependencies_satisfied(self, dependencies: List[str]) -> bool:
        """检查依赖关系是否满足"""
        for dep_id in dependencies:
            if dep_id in self.tasks and self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def get_task_summary(self) -> Dict[str, Any]:
        """获取任务摘要"""
        status_counts = {}
        type_counts = {}
        
        for task in self.tasks.values():
            # 统计状态
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # 统计类型
            task_type = task.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "avg_completion": sum(task.completion_percentage for task in self.tasks.values()) / len(self.tasks) if self.tasks else 0
        }

class EnhancedAIAgent:
    """增强的AI代理"""
    
    def __init__(self, llm, tools, google_api_key: str):
        self.llm = llm
        self.tools = tools
        self.memory_system = EnhancedMemorySystem()
        self.task_planner = EnhancedTaskPlanner()
        self.prompt_generator = IntelligentPromptGenerator()
        self.context_window = []  # 上下文窗口
        self.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 推理能力增强
        self.reasoning_depth = 3  # 推理深度
        self.confidence_threshold = 0.7  # 置信度阈值
        self.learning_rate = 0.1  # 学习率
        
        # 初始化系统记忆
        self._initialize_system_memory()
        
        logger.info(f"Enhanced AI Agent initialized with session ID: {self.session_id}")
    
    def _initialize_system_memory(self):
        """初始化系统记忆"""
        system_knowledge = [
            {
                "content": "我是AutoML Workflow Agent，专门帮助用户完成机器学习任务",
                "importance": 1.0,
                "tags": ["system", "identity"]
            },
            {
                "content": "我可以执行Python代码、分析数据、训练模型、创建可视化",
                "importance": 0.9,
                "tags": ["capabilities", "functions"]
            },
            {
                "content": "安全是最重要的，所有代码都在Docker沙箱中执行",
                "importance": 1.0,
                "tags": ["security", "docker", "safety"]
            },
            {
                "content": "我会记录所有执行历史和机器学习成果到数据库",
                "importance": 0.8,
                "tags": ["database", "logging", "history"]
            }
        ]
        
        for knowledge in system_knowledge:
            self.memory_system.add_memory(
                content=knowledge["content"],
                memory_type=MemoryType.SEMANTIC,
                importance=knowledge["importance"],
                tags=knowledge["tags"]
            )
    
    def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理用户输入的增强版本"""
        try:
            # 1. 记录用户输入到短期记忆
            self.memory_system.add_memory(
                content=f"用户输入: {user_input}",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.6,
                tags=["user_input", "conversation"]
            )
            
            # 2. 分析用户意图和任务类型
            intent_analysis = self._analyze_user_intent(user_input)
            
            # 3. 检索相关记忆
            relevant_memories = self.memory_system.retrieve_memories(
                query=user_input,
                limit=5
            )
            
            # 4. 创建任务计划（如果需要）
            task_plan = None
            if intent_analysis.get("requires_task_planning", False):
                task_plan = self.task_planner.create_task_plan(
                    description=user_input,
                    task_type=intent_analysis.get("task_type"),
                    priority=intent_analysis.get("priority", 5)
                )
            
            # 5. 构建增强的提示
            enhanced_prompt = self._build_enhanced_prompt(
                user_input=user_input,
                intent_analysis=intent_analysis,
                relevant_memories=relevant_memories,
                task_plan=task_plan,
                context=context
            )
            
            # 6. 生成响应
            response = self._generate_response(enhanced_prompt)
            
            # 7. 记录响应到记忆
            self.memory_system.add_memory(
                content=f"Agent响应: {response.get('content', '')}",
                memory_type=MemoryType.SHORT_TERM,
                importance=0.7,
                tags=["agent_response", "conversation"]
            )
            
            # 8. 更新任务状态（如果有任务）
            if task_plan:
                self.task_planner.update_task_status(
                    task_plan.task_id,
                    TaskStatus.IN_PROGRESS
                )
            
            return {
                "response": response,
                "intent_analysis": intent_analysis,
                "task_plan": task_plan.to_dict() if task_plan else None,
                "relevant_memories": [m.to_dict() for m in relevant_memories],
                "session_id": self.session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}", exc_info=True)
            return {
                "response": {"content": f"处理请求时发生错误: {str(e)}"},
                "error": str(e)
            }
    
    def _analyze_user_intent(self, user_input: str) -> Dict[str, Any]:
        """分析用户意图"""
        intent_analysis = {
            "requires_task_planning": False,
            "task_type": None,
            "priority": 5,
            "complexity": "medium",
            "estimated_duration": 30
        }
        
        user_input_lower = user_input.lower()
        
        # 检查是否需要任务规划
        task_indicators = ["分析", "训练", "建模", "预测", "可视化", "处理", "清洗"]
        if any(indicator in user_input_lower for indicator in task_indicators):
            intent_analysis["requires_task_planning"] = True
        
        # 推断任务类型
        if any(keyword in user_input_lower for keyword in ["探索", "查看", "分析", "统计"]):
            intent_analysis["task_type"] = TaskType.DATA_EXPLORATION
        elif any(keyword in user_input_lower for keyword in ["清洗", "预处理", "处理"]):
            intent_analysis["task_type"] = TaskType.DATA_PREPROCESSING
        elif any(keyword in user_input_lower for keyword in ["特征", "变量"]):
            intent_analysis["task_type"] = TaskType.FEATURE_ENGINEERING
        elif any(keyword in user_input_lower for keyword in ["训练", "模型", "建模"]):
            intent_analysis["task_type"] = TaskType.MODEL_TRAINING
        elif any(keyword in user_input_lower for keyword in ["评估", "测试", "验证"]):
            intent_analysis["task_type"] = TaskType.MODEL_EVALUATION
        elif any(keyword in user_input_lower for keyword in ["可视化", "图表", "画图"]):
            intent_analysis["task_type"] = TaskType.VISUALIZATION
        
        # 推断优先级
        if any(keyword in user_input_lower for keyword in ["紧急", "重要", "急需"]):
            intent_analysis["priority"] = 9
        elif any(keyword in user_input_lower for keyword in ["简单", "快速", "简要"]):
            intent_analysis["priority"] = 3
        
        # 推断复杂度
        if any(keyword in user_input_lower for keyword in ["复杂", "深入", "详细", "全面"]):
            intent_analysis["complexity"] = "high"
            intent_analysis["estimated_duration"] = 90
        elif any(keyword in user_input_lower for keyword in ["简单", "快速", "基本"]):
            intent_analysis["complexity"] = "low"
            intent_analysis["estimated_duration"] = 15
        
        return intent_analysis
    
    def _build_enhanced_prompt(self, user_input: str, intent_analysis: Dict[str, Any],
                              relevant_memories: List[MemoryEntry], task_plan: TaskPlan = None,
                              context: Dict[str, Any] = None) -> str:
        """构建增强的提示"""
        prompt_parts = []
        
        # 系统身份
        prompt_parts.append("你是一个增强型AutoML Workflow Agent，具备以下能力：")
        prompt_parts.append("- 智能任务规划和分解")
        prompt_parts.append("- 记忆系统和上下文理解")
        prompt_parts.append("- 自适应学习和推理")
        prompt_parts.append("- 全面的机器学习工具生态")
        
        # 相关记忆
        if relevant_memories:
            prompt_parts.append("\n## 相关记忆:")
            for memory in relevant_memories:
                prompt_parts.append(f"- {memory.content} (重要性: {memory.importance})")
        
        # 意图分析
        prompt_parts.append(f"\n## 用户意图分析:")
        prompt_parts.append(f"- 任务类型: {intent_analysis.get('task_type')}")
        prompt_parts.append(f"- 复杂度: {intent_analysis.get('complexity')}")
        prompt_parts.append(f"- 优先级: {intent_analysis.get('priority')}")
        prompt_parts.append(f"- 预估时长: {intent_analysis.get('estimated_duration')}分钟")
        
        # 任务计划
        if task_plan:
            prompt_parts.append(f"\n## 任务计划 ({task_plan.task_id}):")
            prompt_parts.append(f"- 描述: {task_plan.description}")
            prompt_parts.append(f"- 类型: {task_plan.task_type.value}")
            prompt_parts.append(f"- 子任务数量: {len(task_plan.subtasks)}")
            if task_plan.subtasks:
                prompt_parts.append("- 子任务列表:")
                for subtask in task_plan.subtasks:
                    prompt_parts.append(f"  * {subtask.description}")
        
        # 上下文信息
        if context:
            prompt_parts.append(f"\n## 上下文信息:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
        
        # 用户输入
        prompt_parts.append(f"\n## 用户请求:")
        prompt_parts.append(user_input)
        
        # 响应指导
        prompt_parts.append("\n## 响应要求:")
        prompt_parts.append("1. 基于记忆和上下文提供个性化响应")
        prompt_parts.append("2. 如果需要执行代码，请详细说明计划")
        prompt_parts.append("3. 考虑任务的复杂度和用户的经验水平")
        prompt_parts.append("4. 主动提供相关建议和最佳实践")
        prompt_parts.append("5. 确保所有操作的安全性和可靠性")
        
        return "\n".join(prompt_parts)
    
    def _generate_response(self, enhanced_prompt: str) -> Dict[str, Any]:
        """生成响应"""
        try:
            # 使用智能提示生成器创建更好的提示
            intelligent_prompt = self.prompt_generator.generate_intelligent_prompt(
                user_input=enhanced_prompt,
                context={
                    "session_id": self.session_id,
                    "memory_summary": self.memory_system.get_memory_summary(),
                    "task_summary": self.task_planner.get_task_summary()
                }
            )
            
            # 多层推理处理
            reasoning_chain = self._perform_multi_level_reasoning(enhanced_prompt)
            
            # 生成最终响应
            response_content = self._synthesize_final_response(
                prompt=intelligent_prompt,
                reasoning_chain=reasoning_chain
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(reasoning_chain)
            
            # 生成建议
            suggestions = self._generate_contextual_suggestions(enhanced_prompt, reasoning_chain)
            
            return {
                "content": response_content,
                "confidence": confidence,
                "reasoning": reasoning_chain,
                "suggestions": suggestions,
                "prompt_template": self.prompt_generator.infer_template_type(enhanced_prompt).value
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "content": f"生成响应时发生错误: {str(e)}",
                "error": str(e),
                "confidence": 0.0
            }
    
    def _perform_multi_level_reasoning(self, user_input: str) -> List[Dict[str, Any]]:
        """执行多层推理"""
        reasoning_chain = []
        
        try:
            # 第一层：字面理解
            level1 = {
                "level": 1,
                "type": "literal_understanding",
                "analysis": self._analyze_literal_meaning(user_input),
                "confidence": 0.9
            }
            reasoning_chain.append(level1)
            
            # 第二层：上下文推理
            level2 = {
                "level": 2,
                "type": "contextual_reasoning",
                "analysis": self._analyze_contextual_meaning(user_input),
                "confidence": 0.8
            }
            reasoning_chain.append(level2)
            
            # 第三层：深度洞察
            level3 = {
                "level": 3,
                "type": "deep_insights",
                "analysis": self._generate_deep_insights(user_input),
                "confidence": 0.7
            }
            reasoning_chain.append(level3)
            
            # 自适应学习：更新记忆系统
            self._update_learning_system(user_input, reasoning_chain)
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error in multi-level reasoning: {str(e)}", exc_info=True)
            return [{"level": 1, "type": "error", "analysis": str(e), "confidence": 0.1}]
    
    def _analyze_literal_meaning(self, user_input: str) -> Dict[str, Any]:
        """分析字面含义"""
        return {
            "keywords": self._extract_keywords(user_input),
            "intent": self._classify_intent(user_input),
            "complexity": self._assess_complexity(user_input),
            "domain": self._identify_domain(user_input)
        }
    
    def _analyze_contextual_meaning(self, user_input: str) -> Dict[str, Any]:
        """分析上下文含义"""
        relevant_memories = self.memory_system.retrieve_memories(query=user_input, limit=5)
        similar_tasks = self._find_similar_tasks(user_input)
        
        return {
            "historical_context": [m.content for m in relevant_memories],
            "similar_experiences": similar_tasks,
            "user_patterns": self._analyze_user_patterns(),
            "environmental_factors": self._assess_environmental_factors()
        }
    
    def _generate_deep_insights(self, user_input: str) -> Dict[str, Any]:
        """生成深度洞察"""
        return {
            "hidden_requirements": self._identify_hidden_requirements(user_input),
            "potential_challenges": self._predict_challenges(user_input),
            "optimization_opportunities": self._identify_optimizations(user_input),
            "learning_opportunities": self._identify_learning_points(user_input)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取，可以后续使用NLP库增强
        stop_words = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "和", "或", "但", "而", "与"}
        words = text.split()
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        return keywords[:10]  # 返回前10个关键词
    
    def _classify_intent(self, text: str) -> str:
        """分类意图"""
        text_lower = text.lower()
        
        intent_patterns = {
            "数据分析": ["分析", "探索", "查看", "统计", "分布"],
            "模型训练": ["训练", "建模", "预测", "算法", "模型"],
            "可视化": ["图", "画", "可视化", "展示", "图表"],
            "调试": ["错误", "问题", "调试", "修复", "bug"],
            "优化": ["优化", "改进", "提升", "加速", "效率"],
            "解释": ["解释", "说明", "原理", "为什么", "如何"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        
        return "一般查询"
    
    def _assess_complexity(self, text: str) -> str:
        """评估复杂度"""
        if len(text) > 200:
            return "高"
        elif len(text) > 100:
            return "中"
        else:
            return "低"
    
    def _identify_domain(self, text: str) -> str:
        """识别领域"""
        text_lower = text.lower()
        
        domain_keywords = {
            "机器学习": ["模型", "训练", "预测", "特征", "算法"],
            "数据科学": ["数据", "分析", "统计", "可视化", "探索"],
            "深度学习": ["神经网络", "深度", "卷积", "循环", "transformer"],
            "自然语言处理": ["文本", "语言", "词汇", "语义", "nlp"],
            "计算机视觉": ["图像", "视觉", "检测", "识别", "cv"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return "通用"
    
    def _find_similar_tasks(self, user_input: str) -> List[str]:
        """查找相似任务"""
        # 简化实现，可以后续使用向量相似度搜索
        tasks = [task.description for task in self.task_planner.tasks.values()]
        return tasks[:3]  # 返回前3个相似任务
    
    def _analyze_user_patterns(self) -> Dict[str, Any]:
        """分析用户模式"""
        memories = list(self.memory_system.memories.values())
        
        if not memories:
            return {"pattern": "新用户", "preferences": []}
        
        # 分析用户偏好
        tags = []
        for memory in memories:
            tags.extend(memory.tags)
        
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_preferences = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "pattern": "活跃用户" if len(memories) > 10 else "普通用户",
            "preferences": [pref[0] for pref in top_preferences],
            "interaction_frequency": len(memories) / max(1, (datetime.datetime.now() - memories[0].timestamp).days)
        }
    
    def _assess_environmental_factors(self) -> Dict[str, Any]:
        """评估环境因素"""
        return {
            "time_of_day": datetime.datetime.now().hour,
            "session_length": len(self.context_window),
            "memory_load": len(self.memory_system.memories),
            "task_backlog": len([t for t in self.task_planner.tasks.values() if t.status == TaskStatus.PENDING])
        }
    
    def _identify_hidden_requirements(self, user_input: str) -> List[str]:
        """识别隐藏需求"""
        hidden_reqs = []
        
        if "分析" in user_input and "数据" in user_input:
            hidden_reqs.extend([
                "可能需要数据清洗和预处理",
                "应该检查数据质量和完整性",
                "可能需要可视化来展示结果"
            ])
        
        if "模型" in user_input or "训练" in user_input:
            hidden_reqs.extend([
                "需要划分训练集和测试集",
                "应该进行模型评估和验证",
                "可能需要超参数调优"
            ])
        
        return hidden_reqs
    
    def _predict_challenges(self, user_input: str) -> List[str]:
        """预测挑战"""
        challenges = []
        
        if "大数据" in user_input or "大量" in user_input:
            challenges.append("内存和计算资源限制")
        
        if "实时" in user_input:
            challenges.append("延迟和性能要求")
        
        if "准确" in user_input or "精确" in user_input:
            challenges.append("模型精度和过拟合风险")
        
        return challenges or ["常规技术实现挑战"]
    
    def _identify_optimizations(self, user_input: str) -> List[str]:
        """识别优化机会"""
        optimizations = []
        
        if "慢" in user_input or "时间" in user_input:
            optimizations.extend([
                "代码性能优化",
                "并行处理",
                "缓存机制"
            ])
        
        if "效果" in user_input or "准确" in user_input:
            optimizations.extend([
                "特征工程改进",
                "模型集成",
                "超参数优化"
            ])
        
        return optimizations or ["代码质量和可读性提升"]
    
    def _identify_learning_points(self, user_input: str) -> List[str]:
        """识别学习要点"""
        return [
            "理解业务需求和技术约束的平衡",
            "掌握相关工具和库的最佳实践",
            "学习错误处理和异常情况的应对",
            "培养数据科学的思维模式"
        ]
    
    def _calculate_confidence(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        if not reasoning_chain:
            return 0.0
        
        # 基于推理链的平均置信度
        confidences = [step.get("confidence", 0.5) for step in reasoning_chain]
        base_confidence = sum(confidences) / len(confidences)
        
        # 根据记忆系统的丰富程度调整
        memory_factor = min(1.0, len(self.memory_system.memories) / 100)
        
        # 根据任务历史调整
        task_factor = min(1.0, len(self.task_planner.tasks) / 50)
        
        # 综合置信度
        final_confidence = base_confidence * (0.7 + 0.2 * memory_factor + 0.1 * task_factor)
        
        return min(1.0, final_confidence)
    
    def _generate_contextual_suggestions(self, user_input: str, reasoning_chain: List[Dict[str, Any]]) -> List[str]:
        """生成上下文相关的建议"""
        suggestions = []
        
        # 基于推理链生成建议
        for step in reasoning_chain:
            if step.get("type") == "deep_insights":
                insights = step.get("analysis", {})
                if insights.get("optimization_opportunities"):
                    suggestions.extend(insights["optimization_opportunities"][:2])
        
        # 基于用户历史生成建议
        user_patterns = self._analyze_user_patterns()
        if user_patterns.get("preferences"):
            suggestions.append(f"基于您的偏好，建议使用: {', '.join(user_patterns['preferences'][:3])}")
        
        # 基于当前任务状态生成建议
        pending_tasks = self.task_planner.get_next_tasks(limit=3)
        if pending_tasks:
            suggestions.append(f"您还有 {len(pending_tasks)} 个待完成的任务，建议优先处理高优先级任务")
        
        # 通用建议
        if not suggestions:
            suggestions = [
                "建议先进行数据探索以了解数据特征",
                "推荐使用交叉验证来评估模型性能",
                "考虑将结果可视化以便更好地理解"
            ]
        
        return suggestions[:5]  # 最多返回5个建议
    
    def _synthesize_final_response(self, prompt: str, reasoning_chain: List[Dict[str, Any]]) -> str:
        """综合最终响应"""
        # 这里应该调用LLM生成实际响应，暂时返回结构化的响应
        response_parts = []
        
        response_parts.append("基于多层推理分析，我为您提供以下建议：")
        
        # 添加主要洞察
        for step in reasoning_chain:
            if step.get("level") == 1:
                analysis = step.get("analysis", {})
                response_parts.append(f"\n**需求分析**: 您的请求涉及{analysis.get('domain', '通用')}领域的{analysis.get('intent', '查询')}任务")
        
        # 添加具体建议
        response_parts.append("\n**推荐方案**:")
        response_parts.append("1. 首先进行数据质量检查和基础分析")
        response_parts.append("2. 根据数据特点选择合适的处理方法")
        response_parts.append("3. 实施解决方案并监控结果")
        response_parts.append("4. 记录过程和结果用于后续优化")
        
        return "\n".join(response_parts)
    
    def _update_learning_system(self, user_input: str, reasoning_chain: List[Dict[str, Any]]):
        """更新学习系统"""
        # 将推理过程添加到记忆系统
        reasoning_memory = f"推理过程: {user_input} -> {len(reasoning_chain)}层分析"
        self.memory_system.add_memory(
            content=reasoning_memory,
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            tags=["reasoning", "learning", "analysis"]
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取代理状态"""
        return {
            "session_id": self.session_id,
            "memory_summary": self.memory_system.get_memory_summary(),
            "task_summary": self.task_planner.get_task_summary(),
            "context_window_size": len(self.context_window),
            "status": "active"
        }
    
    def export_session_data(self) -> Dict[str, Any]:
        """导出会话数据"""
        return {
            "session_id": self.session_id,
            "memories": [memory.to_dict() for memory in self.memory_system.memories.values()],
            "tasks": [task.to_dict() for task in self.task_planner.tasks.values()],
            "context_window": self.context_window,
            "export_timestamp": datetime.datetime.now().isoformat()
        }
    
    def import_session_data(self, session_data: Dict[str, Any]):
        """导入会话数据"""
        try:
            # 导入记忆
            for memory_data in session_data.get("memories", []):
                memory = MemoryEntry(
                    entry_id=memory_data["entry_id"],
                    memory_type=MemoryType(memory_data["memory_type"]),
                    content=memory_data["content"],
                    importance=memory_data["importance"],
                    timestamp=datetime.datetime.fromisoformat(memory_data["timestamp"]),
                    tags=memory_data["tags"],
                    related_tasks=memory_data["related_tasks"],
                    access_count=memory_data.get("access_count", 0),
                    last_accessed=datetime.datetime.fromisoformat(memory_data["last_accessed"]) if memory_data.get("last_accessed") else None
                )
                self.memory_system.memories[memory.entry_id] = memory
            
            # 导入任务
            for task_data in session_data.get("tasks", []):
                task = TaskPlan(
                    task_id=task_data["task_id"],
                    task_type=TaskType(task_data["task_type"]),
                    description=task_data["description"],
                    status=TaskStatus(task_data["status"]),
                    priority=task_data["priority"],
                    estimated_duration=task_data.get("estimated_duration"),
                    dependencies=task_data["dependencies"],
                    subtasks=[],  # 简化处理，可以后续增强
                    created_at=datetime.datetime.fromisoformat(task_data["created_at"]),
                    updated_at=datetime.datetime.fromisoformat(task_data["updated_at"]),
                    completion_percentage=task_data.get("completion_percentage", 0.0)
                )
                self.task_planner.tasks[task.task_id] = task
            
            # 导入上下文窗口
            self.context_window = session_data.get("context_window", [])
            
            logger.info(f"Successfully imported session data from {session_data.get('session_id')}")
            
        except Exception as e:
            logger.error(f"Error importing session data: {str(e)}", exc_info=True)
            raise