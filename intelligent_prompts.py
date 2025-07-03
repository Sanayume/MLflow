"""
Intelligent Prompt Generation System
智能提示生成系统 - 增强版本
基于gemini-cli的智能提示架构，结合用户意图分析和上下文感知
提供动态提示生成、模板管理和智能推荐
"""

import asyncio
import json
import re
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
import sqlite3
import pickle
import logging

logger = logging.getLogger(__name__)

# 新增智能意图分析类型
class IntentType(Enum):
    """用户意图类型"""
    DATA_EXPLORATION = "data_exploration"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    FEATURE_ENGINEERING = "feature_engineering"
    VISUALIZATION = "visualization"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"
    GENERAL_QUESTION = "general_question"

class PromptComplexity(Enum):
    """提示复杂度"""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class UserIntent:
    """用户意图分析结果"""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[Tuple[IntentType, float]] = field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    complexity_level: PromptComplexity = PromptComplexity.INTERMEDIATE
    context_keywords: Set[str] = field(default_factory=set)

@dataclass
class PromptTemplateData:
    """智能提示模板数据"""
    id: str
    name: str
    description: str
    template_text: str
    intent_types: List[IntentType]
    complexity_level: PromptComplexity
    variables: List[str]  # 模板变量
    tags: Set[str] = field(default_factory=set)
    usage_count: int = 0
    effectiveness_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

@dataclass
class PromptGenerationContext:
    """提示生成上下文"""
    user_input: str
    ml_context: Dict[str, Any]  # 来自环境分析器的上下文
    conversation_history: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    available_tools: List[str] = field(default_factory=list)
    data_context: Dict[str, Any] = field(default_factory=dict)
    model_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedPrompt:
    """生成的提示"""
    id: str
    original_input: str
    generated_text: str
    intent: UserIntent
    template_used: Optional[str]
    confidence: float
    reasoning: str
    suggestions: List[str] = field(default_factory=list)
    context_used: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

class IntelligentMLPromptGenerator:
    """智能ML提示生成器 - 增强版本"""
    
    def __init__(self, project_root: str = ".", prompt_db_path: str = None):
        self.project_root = Path(project_root).resolve()
        self.prompt_db_path = prompt_db_path or self.project_root / ".mlagent" / "prompts.db"
        
        # 确保目录存在
        self.prompt_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 内存中的数据结构
        self.templates: Dict[str, PromptTemplateData] = {}
        self.generated_prompts: Dict[str, GeneratedPrompt] = {}
        
        # 缓存
        self.generation_cache = {}
        self.cache_ttl = 1800  # 30分钟
        
        # 初始化数据库
        self._init_database()
        
        # 加载预定义模板
        asyncio.create_task(self._load_predefined_templates())
        
        # 加载现有数据
        asyncio.create_task(self._load_data())
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.conn = sqlite3.connect(str(self.prompt_db_path), check_same_thread=False)
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                template_text TEXT NOT NULL,
                intent_types TEXT NOT NULL,
                complexity_level TEXT NOT NULL,
                variables TEXT NOT NULL,
                tags TEXT,
                usage_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.5,
                created_at TEXT,
                last_used TEXT
            );
            
            CREATE TABLE IF NOT EXISTS generated_prompts (
                id TEXT PRIMARY KEY,
                original_input TEXT NOT NULL,
                generated_text TEXT NOT NULL,
                intent_data TEXT NOT NULL,
                template_used TEXT,
                confidence REAL,
                reasoning TEXT,
                suggestions TEXT,
                context_used TEXT,
                generated_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                feedback_text TEXT,
                timestamp TEXT,
                FOREIGN KEY (prompt_id) REFERENCES generated_prompts (id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_template_intent ON prompt_templates(intent_types);
            CREATE INDEX IF NOT EXISTS idx_template_complexity ON prompt_templates(complexity_level);
            CREATE INDEX IF NOT EXISTS idx_prompt_timestamp ON generated_prompts(generated_at);
        """)
        
        self.conn.commit()
    
    async def _load_predefined_templates(self):
        """加载预定义的提示模板"""
        print("📝 加载预定义提示模板...")
        
        predefined_templates = [
            {
                "name": "数据探索分析",
                "description": "用于数据集初步探索和统计分析",
                "template_text": "请帮我分析数据集 {dataset_name}。我想了解：\n1. 数据的基本统计信息\n2. 数据质量（缺失值、异常值）\n3. 特征之间的相关性\n4. 数据分布特征\n请提供详细的分析和可视化建议。",
                "intent_types": [IntentType.DATA_EXPLORATION],
                "complexity_level": PromptComplexity.INTERMEDIATE,
                "variables": ["dataset_name"],
                "tags": {"exploration", "statistics", "visualization"}
            },
            {
                "name": "模型训练指导",
                "description": "用于机器学习模型训练的指导",
                "template_text": "我想训练一个 {model_type} 模型来解决 {problem_type} 问题。\n数据集：{dataset_info}\n目标：{target_description}\n请帮我：\n1. 选择合适的算法和参数\n2. 设计训练流程\n3. 定义评估指标\n4. 提供代码实现建议",
                "intent_types": [IntentType.MODEL_TRAINING],
                "complexity_level": PromptComplexity.ADVANCED,
                "variables": ["model_type", "problem_type", "dataset_info", "target_description"],
                "tags": {"training", "algorithm", "parameters"}
            },
            {
                "name": "特征工程优化",
                "description": "用于特征工程和数据预处理",
                "template_text": "针对 {data_type} 数据，我需要进行特征工程。\n当前特征：{current_features}\n目标：{objective}\n请建议：\n1. 特征选择策略\n2. 特征变换方法\n3. 新特征创建思路\n4. 特征重要性评估方法",
                "intent_types": [IntentType.FEATURE_ENGINEERING],
                "complexity_level": PromptComplexity.INTERMEDIATE,
                "variables": ["data_type", "current_features", "objective"],
                "tags": {"feature_engineering", "preprocessing", "selection"}
            },
            {
                "name": "模型评估诊断",
                "description": "用于模型性能评估和诊断",
                "template_text": "我的 {model_name} 模型在 {dataset} 上的表现是：\n{performance_metrics}\n请帮我：\n1. 分析模型性能\n2. 诊断可能的问题\n3. 提出改进建议\n4. 推荐进一步的评估方法",
                "intent_types": [IntentType.MODEL_EVALUATION],
                "complexity_level": PromptComplexity.INTERMEDIATE,
                "variables": ["model_name", "dataset", "performance_metrics"],
                "tags": {"evaluation", "metrics", "diagnosis"}
            },
            {
                "name": "代码调试帮助",
                "description": "用于ML代码的调试和错误解决",
                "template_text": "我在运行ML代码时遇到了问题：\n错误信息：{error_message}\n代码片段：{code_snippet}\n环境信息：{environment_info}\n请帮我：\n1. 分析错误原因\n2. 提供解决方案\n3. 给出修复后的代码\n4. 预防类似问题的建议",
                "intent_types": [IntentType.DEBUGGING],
                "complexity_level": PromptComplexity.SIMPLE,
                "variables": ["error_message", "code_snippet", "environment_info"],
                "tags": {"debugging", "error", "troubleshooting"}
            },
            {
                "name": "可视化设计",
                "description": "用于数据可视化和结果展示",
                "template_text": "我需要为 {data_description} 创建可视化。\n目的：{visualization_purpose}\n目标受众：{target_audience}\n请推荐：\n1. 合适的图表类型\n2. 可视化库和工具\n3. 设计原则和最佳实践\n4. 具体的实现代码",
                "intent_types": [IntentType.VISUALIZATION],
                "complexity_level": PromptComplexity.INTERMEDIATE,
                "variables": ["data_description", "visualization_purpose", "target_audience"],
                "tags": {"visualization", "charts", "presentation"}
            }
        ]
        
        for template_data in predefined_templates:
            template_id = hashlib.md5(template_data["name"].encode()).hexdigest()[:12]
            
            template = PromptTemplateData(
                id=template_id,
                name=template_data["name"],
                description=template_data["description"],
                template_text=template_data["template_text"],
                intent_types=template_data["intent_types"],
                complexity_level=template_data["complexity_level"],
                variables=template_data["variables"],
                tags=template_data["tags"]
            )
            
            self.templates[template_id] = template
            await self._save_template(template)
        
        print(f"✅ 加载了 {len(predefined_templates)} 个预定义模板")
    
    async def _load_data(self):
        """从数据库加载数据"""
        print("💾 加载现有提示数据...")
        
        cursor = self.conn.cursor()
        
        # 加载模板
        cursor.execute("SELECT * FROM prompt_templates")
        for row in cursor.fetchall():
            template = self._row_to_template(row)
            self.templates[template.id] = template
        
        # 加载生成的提示
        cursor.execute("SELECT * FROM generated_prompts")
        for row in cursor.fetchall():
            prompt = self._row_to_generated_prompt(row)
            self.generated_prompts[prompt.id] = prompt
        
        print(f"✅ 加载完成: {len(self.templates)} 个模板, {len(self.generated_prompts)} 个生成的提示")
    
    def _row_to_template(self, row) -> PromptTemplateData:
        """将数据库行转换为模板对象"""
        return PromptTemplateData(
            id=row[0],
            name=row[1],
            description=row[2],
            template_text=row[3],
            intent_types=[IntentType(t) for t in json.loads(row[4])],
            complexity_level=PromptComplexity(row[5]),
            variables=json.loads(row[6]),
            tags=set(json.loads(row[7]) if row[7] else []),
            usage_count=row[8],
            effectiveness_score=row[9],
            created_at=datetime.fromisoformat(row[10]),
            last_used=datetime.fromisoformat(row[11]) if row[11] else None
        )
    
    def _row_to_generated_prompt(self, row) -> GeneratedPrompt:
        """将数据库行转换为生成提示对象"""
        intent_data = json.loads(row[3])
        intent = UserIntent(
            primary_intent=IntentType(intent_data['primary_intent']),
            confidence=intent_data['confidence'],
            secondary_intents=[(IntentType(t), s) for t, s in intent_data.get('secondary_intents', [])],
            extracted_entities=intent_data.get('extracted_entities', {}),
            complexity_level=PromptComplexity(intent_data.get('complexity_level', 'intermediate')),
            context_keywords=set(intent_data.get('context_keywords', []))
        )
        
        return GeneratedPrompt(
            id=row[0],
            original_input=row[1],
            generated_text=row[2],
            intent=intent,
            template_used=row[4],
            confidence=row[5],
            reasoning=row[6],
            suggestions=json.loads(row[7]) if row[7] else [],
            context_used=json.loads(row[8]) if row[8] else {},
            generated_at=datetime.fromisoformat(row[9])
        )
    
    async def generate_prompt(self, 
                            user_input: str,
                            context: PromptGenerationContext = None) -> GeneratedPrompt:
        """主要入口：智能生成提示"""
        
        # 生成缓存键
        cache_key = hashlib.md5(f"{user_input}_{context}".encode()).hexdigest()
        
        # 检查缓存
        if cache_key in self.generation_cache:
            cached_result, timestamp = self.generation_cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                print("📊 使用缓存的提示生成结果")
                return cached_result
        
        print(f"🧠 开始智能提示生成: {user_input[:50]}...")
        
        try:
            # 1. 分析用户意图
            intent = await self._analyze_user_intent(user_input, context)
            
            # 2. 选择最佳模板
            best_template = await self._select_best_template(intent, context)
            
            # 3. 生成提示文本
            generated_text, reasoning = await self._generate_prompt_text(
                user_input, intent, best_template, context
            )
            
            # 4. 生成智能建议
            suggestions = await self._generate_suggestions(intent, context)
            
            # 5. 计算置信度
            confidence = self._calculate_generation_confidence(intent, best_template, context)
            
            # 6. 创建生成的提示对象
            prompt_id = hashlib.md5(f"{user_input}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            generated_prompt = GeneratedPrompt(
                id=prompt_id,
                original_input=user_input,
                generated_text=generated_text,
                intent=intent,
                template_used=best_template.id if best_template else None,
                confidence=confidence,
                reasoning=reasoning,
                suggestions=suggestions,
                context_used=asdict(context) if context else {}
            )
            
            # 7. 保存到数据库
            self.generated_prompts[prompt_id] = generated_prompt
            await self._save_generated_prompt(generated_prompt)
            
            # 8. 更新模板使用统计
            if best_template:
                await self._update_template_usage(best_template.id)
            
            # 9. 缓存结果
            self.generation_cache[cache_key] = (generated_prompt, datetime.now().timestamp())
            
            print(f"✅ 提示生成完成，置信度: {confidence:.2f}")
            return generated_prompt
            
        except Exception as e:
            print(f"❌ 提示生成失败: {str(e)}")
            raise
    
    async def _analyze_user_intent(self, 
                                 user_input: str, 
                                 context: PromptGenerationContext = None) -> UserIntent:
        """分析用户意图"""
        
        # 关键词模式匹配
        intent_patterns = {
            IntentType.DATA_EXPLORATION: [
                r'探索|分析|查看|检查|统计|描述|概览',
                r'数据集|数据|data|dataset|explore|analyze',
                r'分布|相关性|缺失值|异常值|质量'
            ],
            IntentType.MODEL_TRAINING: [
                r'训练|建模|拟合|学习|train|fit|model',
                r'算法|分类|回归|聚类|深度学习|神经网络',
                r'参数|超参数|优化|调优'
            ],
            IntentType.MODEL_EVALUATION: [
                r'评估|测试|验证|性能|准确率|evaluate',
                r'指标|metrics|score|accuracy|precision|recall',
                r'交叉验证|验证集|测试集'
            ],
            IntentType.FEATURE_ENGINEERING: [
                r'特征|feature|engineering|变换|选择',
                r'预处理|归一化|标准化|编码|降维',
                r'特征重要性|特征选择|特征提取'
            ],
            IntentType.VISUALIZATION: [
                r'可视化|图表|画图|plot|chart|visualize',
                r'柱状图|散点图|热力图|折线图|直方图',
                r'matplotlib|seaborn|plotly'
            ],
            IntentType.DEBUGGING: [
                r'错误|报错|debug|fix|解决|问题',
                r'异常|exception|error|bug|调试',
                r'不工作|失败|无法运行'
            ],
            IntentType.OPTIMIZATION: [
                r'优化|提升|改进|accelerate|optimize',
                r'性能|速度|内存|效率|performance',
                r'并行|分布式|GPU|加速'
            ],
            IntentType.DEPLOYMENT: [
                r'部署|deploy|生产|production|服务',
                r'API|web|应用|application|上线',
                r'docker|kubernetes|云|cloud'
            ]
        }
        
        # 计算每个意图的得分
        intent_scores = {}
        text_lower = user_input.lower()
        
        for intent_type, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent_type] = score / len(patterns)
        
        # 确定主要意图
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            primary_confidence = intent_scores[primary_intent]
            
            # 归一化置信度
            total_score = sum(intent_scores.values())
            primary_confidence = primary_confidence / total_score if total_score > 0 else 0.1
            
            # 确定次要意图
            secondary_intents = [
                (intent, score / total_score) 
                for intent, score in intent_scores.items() 
                if intent != primary_intent and score > 0
            ]
            secondary_intents.sort(key=lambda x: x[1], reverse=True)
            
        else:
            primary_intent = IntentType.GENERAL_QUESTION
            primary_confidence = 0.1
            secondary_intents = []
        
        # 提取实体
        entities = self._extract_entities(user_input)
        
        # 确定复杂度
        complexity = self._determine_complexity(user_input, context)
        
        # 提取上下文关键词
        context_keywords = self._extract_context_keywords(user_input, context)
        
        return UserIntent(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            secondary_intents=secondary_intents[:3],  # 最多3个次要意图
            extracted_entities=entities,
            complexity_level=complexity,
            context_keywords=context_keywords
        )
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体信息"""
        entities = {
            'datasets': [],
            'models': [],
            'algorithms': [],
            'metrics': [],
            'libraries': [],
            'file_paths': []
        }
        
        # 数据集模式
        dataset_patterns = r'(\w+\.csv|\w+\.json|\w+\.parquet|\w+\.xlsx|\w+_data|\w+_dataset)'
        entities['datasets'] = re.findall(dataset_patterns, text.lower())
        
        # 模型算法
        algorithm_keywords = [
            'random forest', 'svm', 'linear regression', 'logistic regression',
            'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
            'xgboost', 'lightgbm', 'catboost', 'decision tree'
        ]
        for keyword in algorithm_keywords:
            if keyword in text.lower():
                entities['algorithms'].append(keyword)
        
        # 评估指标
        metric_keywords = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc',
            'mse', 'mae', 'rmse', 'r2', 'loss'
        ]
        for keyword in metric_keywords:
            if keyword in text.lower():
                entities['metrics'].append(keyword)
        
        # 库名
        library_keywords = [
            'pandas', 'numpy', 'sklearn', 'tensorflow', 'pytorch',
            'matplotlib', 'seaborn', 'plotly', 'xgboost'
        ]
        for keyword in library_keywords:
            if keyword in text.lower():
                entities['libraries'].append(keyword)
        
        # 文件路径
        file_path_pattern = r'([./][\w/.-]+\.\w+)'
        entities['file_paths'] = re.findall(file_path_pattern, text)
        
        return entities
    
    def _determine_complexity(self, 
                            user_input: str, 
                            context: PromptGenerationContext = None) -> PromptComplexity:
        """确定提示复杂度"""
        
        complexity_indicators = {
            PromptComplexity.SIMPLE: [
                r'简单|基础|basic|simple|入门',
                r'怎么|如何|what|how',
                r'开始|start|begin'
            ],
            PromptComplexity.ADVANCED: [
                r'高级|advanced|复杂|complex',
                r'优化|optimization|性能|performance',
                r'分布式|distributed|并行|parallel',
                r'深度|deep|神经网络|neural'
            ],
            PromptComplexity.EXPERT: [
                r'专家|expert|研究|research',
                r'论文|paper|算法实现|implementation',
                r'自定义|custom|底层|low-level'
            ]
        }
        
        text_lower = user_input.lower()
        scores = {}
        
        for complexity, patterns in complexity_indicators.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            if score > 0:
                scores[complexity] = score
        
        # 考虑输入长度
        if len(user_input) > 200:
            scores[PromptComplexity.ADVANCED] = scores.get(PromptComplexity.ADVANCED, 0) + 1
        elif len(user_input) < 50:
            scores[PromptComplexity.SIMPLE] = scores.get(PromptComplexity.SIMPLE, 0) + 1
        
        # 考虑上下文
        if context and context.ml_context:
            if len(context.ml_context.get('existing_models', [])) > 3:
                scores[PromptComplexity.ADVANCED] = scores.get(PromptComplexity.ADVANCED, 0) + 1
        
        if scores:
            return max(scores, key=scores.get)
        else:
            return PromptComplexity.INTERMEDIATE
    
    def _extract_context_keywords(self, 
                                user_input: str, 
                                context: PromptGenerationContext = None) -> Set[str]:
        """提取上下文关键词"""
        keywords = set()
        
        # 从用户输入提取
        words = re.findall(r'\b\w+\b', user_input.lower())
        ml_related_words = {
            'model', 'data', 'train', 'test', 'feature', 'accuracy',
            'prediction', 'classification', 'regression', 'clustering'
        }
        keywords.update(word for word in words if word in ml_related_words)
        
        # 从上下文提取
        if context:
            if context.current_task:
                keywords.add(context.current_task.lower())
            
            keywords.update(context.available_tools)
            
            if context.data_context:
                keywords.update(str(v).lower() for v in context.data_context.values() if isinstance(v, str))
        
        return keywords
    
    async def _select_best_template(self, 
                                  intent: UserIntent, 
                                  context: PromptGenerationContext = None) -> Optional[PromptTemplateData]:
        """选择最佳模板"""
        
        if not self.templates:
            return None
        
        # 筛选匹配意图的模板
        matching_templates = [
            template for template in self.templates.values()
            if intent.primary_intent in template.intent_types
        ]
        
        if not matching_templates:
            # 如果没有直接匹配的，寻找次要意图匹配
            for secondary_intent, _ in intent.secondary_intents:
                matching_templates = [
                    template for template in self.templates.values()
                    if secondary_intent in template.intent_types
                ]
                if matching_templates:
                    break
        
        if not matching_templates:
            return None
        
        # 按多个因素评分
        scored_templates = []
        for template in matching_templates:
            score = 0
            
            # 意图匹配度
            if intent.primary_intent in template.intent_types:
                score += 3
            
            # 复杂度匹配度
            if template.complexity_level == intent.complexity_level:
                score += 2
            elif abs(list(PromptComplexity).index(template.complexity_level) - list(PromptComplexity).index(intent.complexity_level)) <= 1:
                score += 1
            
            # 使用频率和效果
            score += template.usage_count * 0.1
            score += template.effectiveness_score * 2
            
            # 标签匹配度
            if template.tags & intent.context_keywords:
                score += len(template.tags & intent.context_keywords)
            
            scored_templates.append((template, score))
        
        # 选择得分最高的模板
        scored_templates.sort(key=lambda x: x[1], reverse=True)
        return scored_templates[0][0] if scored_templates else None
    
    async def _generate_prompt_text(self, 
                                  user_input: str,
                                  intent: UserIntent,
                                  template: Optional[PromptTemplateData],
                                  context: PromptGenerationContext = None) -> Tuple[str, str]:
        """生成提示文本"""
        
        if template:
            # 使用模板生成
            prompt_text, reasoning = await self._generate_from_template(
                user_input, intent, template, context
            )
        else:
            # 直接基于意图生成
            prompt_text, reasoning = await self._generate_from_intent(
                user_input, intent, context
            )
        
        # 添加上下文信息
        if context and context.ml_context:
            prompt_text = await self._enhance_with_context(prompt_text, context)
        
        return prompt_text, reasoning
    
    async def _generate_from_template(self, 
                                    user_input: str,
                                    intent: UserIntent,
                                    template: PromptTemplateData,
                                    context: PromptGenerationContext = None) -> Tuple[str, str]:
        """从模板生成提示"""
        
        # 提取模板变量的值
        variable_values = await self._extract_template_variables(
            user_input, intent, template, context
        )
        
        # 替换模板变量
        prompt_text = template.template_text
        for var, value in variable_values.items():
            placeholder = "{" + var + "}"
            prompt_text = prompt_text.replace(placeholder, str(value))
        
        # 如果还有未替换的变量，用占位符替换
        remaining_vars = re.findall(r'\{(\w+)\}', prompt_text)
        for var in remaining_vars:
            placeholder = "{" + var + "}"
            prompt_text = prompt_text.replace(placeholder, f"[请填写{var}]")
        
        reasoning = f"使用模板 '{template.name}' 生成提示，匹配意图 {intent.primary_intent.value}"
        
        return prompt_text, reasoning
    
    async def _generate_from_intent(self, 
                                  user_input: str,
                                  intent: UserIntent,
                                  context: PromptGenerationContext = None) -> Tuple[str, str]:
        """基于意图直接生成提示"""
        
        intent_templates = {
            IntentType.DATA_EXPLORATION: "请帮我分析和探索数据。我想了解数据的基本特征、质量和分布情况。",
            IntentType.MODEL_TRAINING: "请指导我训练机器学习模型。包括算法选择、参数设置和训练流程。",
            IntentType.MODEL_EVALUATION: "请帮我评估模型性能。分析评估指标，诊断问题，提出改进建议。",
            IntentType.FEATURE_ENGINEERING: "请协助我进行特征工程。包括特征选择、变换和创建。",
            IntentType.VISUALIZATION: "请帮我创建数据可视化。选择合适的图表类型和设计方案。",
            IntentType.DEBUGGING: "请帮我解决代码问题。分析错误原因并提供解决方案。",
            IntentType.OPTIMIZATION: "请帮我优化ML系统性能。提升速度、效率和资源利用率。",
            IntentType.DEPLOYMENT: "请指导我部署ML模型到生产环境。包括服务化和监控。",
            IntentType.GENERAL_QUESTION: "请回答我的ML相关问题。提供详细和实用的建议。"
        }
        
        base_prompt = intent_templates.get(intent.primary_intent, intent_templates[IntentType.GENERAL_QUESTION])
        
        # 添加用户输入的具体内容
        prompt_text = f"{base_prompt}\n\n用户需求：{user_input}"
        
        # 根据提取的实体添加更多上下文
        if intent.extracted_entities:
            context_info = []
            for entity_type, entities in intent.extracted_entities.items():
                if entities:
                    context_info.append(f"{entity_type}: {', '.join(entities)}")
            
            if context_info:
                prompt_text += f"\n\n相关信息：\n" + "\n".join(context_info)
        
        reasoning = f"基于意图 {intent.primary_intent.value} 直接生成提示，置信度 {intent.confidence:.2f}"
        
        return prompt_text, reasoning
    
    async def _extract_template_variables(self, 
                                        user_input: str,
                                        intent: UserIntent,
                                        template: PromptTemplateData,
                                        context: PromptGenerationContext = None) -> Dict[str, str]:
        """提取模板变量的值"""
        
        variable_values = {}
        
        for var in template.variables:
            value = None
            
            # 从实体中提取
            if var in ['dataset_name', 'dataset']:
                if intent.extracted_entities.get('datasets'):
                    value = intent.extracted_entities['datasets'][0]
                elif context and context.data_context.get('current_dataset'):
                    value = context.data_context['current_dataset']
            
            elif var in ['model_type', 'model_name']:
                if intent.extracted_entities.get('algorithms'):
                    value = intent.extracted_entities['algorithms'][0]
                elif intent.extracted_entities.get('models'):
                    value = intent.extracted_entities['models'][0]
            
            elif var in ['error_message']:
                # 尝试从用户输入中提取错误信息
                error_pattern = r'error[:\s]*([^.!?\n]+)'
                match = re.search(error_pattern, user_input, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
            
            # 从上下文中提取
            if not value and context:
                if var == 'current_task' and context.current_task:
                    value = context.current_task
                elif var in context.data_context:
                    value = str(context.data_context[var])
                elif var in context.model_context:
                    value = str(context.model_context[var])
            
            # 使用默认值或占位符
            if not value:
                value = f"[{var}]"
            
            variable_values[var] = value
        
        return variable_values
    
    async def _enhance_with_context(self, 
                                  prompt_text: str, 
                                  context: PromptGenerationContext) -> str:
        """使用上下文增强提示"""
        
        context_additions = []
        
        # 添加环境信息
        if context.ml_context:
            ml_ctx = context.ml_context
            
            if ml_ctx.get('available_datasets'):
                datasets = ml_ctx['available_datasets']
                if len(datasets) > 0:
                    context_additions.append(f"可用数据集: {len(datasets)} 个")
            
            if ml_ctx.get('existing_models'):
                models = ml_ctx['existing_models']
                if len(models) > 0:
                    context_additions.append(f"现有模型: {len(models)} 个")
            
            if ml_ctx.get('ml_frameworks'):
                frameworks = ml_ctx['ml_frameworks']
                if frameworks:
                    context_additions.append(f"ML框架: {', '.join(frameworks[:3])}")
        
        # 添加工具信息
        if context.available_tools:
            context_additions.append(f"可用工具: {', '.join(context.available_tools[:5])}")
        
        # 添加对话历史
        if context.conversation_history:
            recent_history = context.conversation_history[-2:]  # 最近2条
            if recent_history:
                context_additions.append(f"最近对话: {'; '.join(recent_history)}")
        
        if context_additions:
            context_text = "\n\n环境上下文:\n" + "\n".join(f"- {addition}" for addition in context_additions)
            prompt_text += context_text
        
        return prompt_text
    
    async def _generate_suggestions(self, 
                                  intent: UserIntent, 
                                  context: PromptGenerationContext = None) -> List[str]:
        """生成智能建议"""
        
        suggestions = []
        
        # 基于意图的建议
        intent_suggestions = {
            IntentType.DATA_EXPLORATION: [
                "建议先查看数据的基本统计信息",
                "检查数据质量，特别是缺失值和异常值",
                "创建数据可视化来理解分布特征"
            ],
            IntentType.MODEL_TRAINING: [
                "建议先进行数据预处理和特征工程",
                "使用交叉验证来评估模型性能",
                "尝试多种算法并比较结果"
            ],
            IntentType.MODEL_EVALUATION: [
                "使用多种评估指标来全面评估模型",
                "分析混淆矩阵来理解分类错误",
                "检查模型在不同数据子集上的表现"
            ],
            IntentType.FEATURE_ENGINEERING: [
                "分析特征重要性来指导特征选择",
                "考虑创建交互特征和多项式特征",
                "使用领域知识来设计有意义的特征"
            ],
            IntentType.VISUALIZATION: [
                "选择适合数据类型和目的的图表",
                "确保可视化清晰易懂",
                "考虑交互式可视化来探索数据"
            ],
            IntentType.DEBUGGING: [
                "仔细阅读错误信息和堆栈跟踪",
                "检查数据类型和形状是否匹配",
                "使用调试工具逐步检查代码"
            ]
        }
        
        base_suggestions = intent_suggestions.get(intent.primary_intent, [])
        suggestions.extend(base_suggestions[:3])  # 最多3个基础建议
        
        # 基于上下文的建议
        if context:
            if context.ml_context:
                ml_ctx = context.ml_context
                
                # 数据相关建议
                if ml_ctx.get('data_volume_gb', 0) > 5:
                    suggestions.append("数据量较大，考虑使用采样或分布式处理")
                
                # GPU建议
                if not ml_ctx.get('gpu_available') and 'tensorflow' in ml_ctx.get('ml_frameworks', []):
                    suggestions.append("考虑使用GPU加速深度学习训练")
                
                # 模型建议
                if len(ml_ctx.get('existing_models', [])) > 0:
                    suggestions.append("可以参考现有模型的架构和参数")
        
        return suggestions[:5]  # 最多返回5个建议
    
    def _calculate_generation_confidence(self, 
                                       intent: UserIntent,
                                       template: Optional[PromptTemplateData],
                                       context: PromptGenerationContext = None) -> float:
        """计算生成置信度"""
        
        confidence = intent.confidence  # 基础置信度来自意图分析
        
        # 模板匹配加分
        if template:
            confidence += 0.2
            
            # 模板效果加分
            confidence += template.effectiveness_score * 0.1
            
            # 变量填充完整性
            if template.variables:
                filled_vars = sum(1 for var in template.variables if f"[{var}]" not in template.template_text)
                completeness = filled_vars / len(template.variables)
                confidence += completeness * 0.1
        
        # 上下文丰富度加分
        if context:
            if context.ml_context:
                confidence += 0.1
            if context.conversation_history:
                confidence += 0.05
            if context.available_tools:
                confidence += 0.05
        
        # 实体提取质量加分
        if intent.extracted_entities:
            entity_count = sum(len(entities) for entities in intent.extracted_entities.values())
            confidence += min(0.1, entity_count * 0.02)
        
        return min(1.0, confidence)
    
    async def _save_template(self, template: PromptTemplateData):
        """保存模板到数据库"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO prompt_templates 
            (id, name, description, template_text, intent_types, complexity_level, 
             variables, tags, usage_count, effectiveness_score, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            template.id,
            template.name,
            template.description,
            template.template_text,
            json.dumps([intent.value for intent in template.intent_types]),
            template.complexity_level.value,
            json.dumps(template.variables),
            json.dumps(list(template.tags)),
            template.usage_count,
            template.effectiveness_score,
            template.created_at.isoformat(),
            template.last_used.isoformat() if template.last_used else None
        ))
        
        self.conn.commit()
    
    async def _save_generated_prompt(self, prompt: GeneratedPrompt):
        """保存生成的提示到数据库"""
        cursor = self.conn.cursor()
        
        intent_data = {
            'primary_intent': prompt.intent.primary_intent.value,
            'confidence': prompt.intent.confidence,
            'secondary_intents': [(intent.value, score) for intent, score in prompt.intent.secondary_intents],
            'extracted_entities': prompt.intent.extracted_entities,
            'complexity_level': prompt.intent.complexity_level.value,
            'context_keywords': list(prompt.intent.context_keywords)
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO generated_prompts 
            (id, original_input, generated_text, intent_data, template_used, 
             confidence, reasoning, suggestions, context_used, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt.id,
            prompt.original_input,
            prompt.generated_text,
            json.dumps(intent_data),
            prompt.template_used,
            prompt.confidence,
            prompt.reasoning,
            json.dumps(prompt.suggestions),
            json.dumps(prompt.context_used),
            prompt.generated_at.isoformat()
        ))
        
        self.conn.commit()
    
    async def _update_template_usage(self, template_id: str):
        """更新模板使用统计"""
        if template_id in self.templates:
            template = self.templates[template_id]
            template.usage_count += 1
            template.last_used = datetime.now()
            await self._save_template(template)
    
    async def provide_feedback(self, prompt_id: str, rating: int, feedback_text: str = None):
        """提供用户反馈"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_feedback (prompt_id, rating, feedback_text, timestamp)
            VALUES (?, ?, ?, ?)
        """, (prompt_id, rating, feedback_text, datetime.now().isoformat()))
        
        self.conn.commit()
        
        # 更新模板效果评分
        if prompt_id in self.generated_prompts:
            generated_prompt = self.generated_prompts[prompt_id]
            if generated_prompt.template_used:
                await self._update_template_effectiveness(generated_prompt.template_used, rating)
        
        print(f"✅ 反馈已记录: 评分 {rating}/5")
    
    async def _update_template_effectiveness(self, template_id: str, rating: int):
        """更新模板效果评分"""
        if template_id in self.templates:
            template = self.templates[template_id]
            
            # 简单的移动平均
            normalized_rating = rating / 5.0  # 转换为0-1范围
            template.effectiveness_score = (template.effectiveness_score * 0.8 + normalized_rating * 0.2)
            
            await self._save_template(template)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取提示生成统计"""
        if not self.generated_prompts:
            return {'total_prompts': 0}
        
        # 意图分布
        intent_counts = {}
        for prompt in self.generated_prompts.values():
            intent = prompt.intent.primary_intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # 复杂度分布
        complexity_counts = {}
        for prompt in self.generated_prompts.values():
            complexity = prompt.intent.complexity_level.value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # 平均置信度
        avg_confidence = sum(p.confidence for p in self.generated_prompts.values()) / len(self.generated_prompts)
        
        # 模板使用统计
        template_usage = {}
        for prompt in self.generated_prompts.values():
            if prompt.template_used:
                template_name = self.templates[prompt.template_used].name
                template_usage[template_name] = template_usage.get(template_name, 0) + 1
        
        return {
            'total_prompts': len(self.generated_prompts),
            'total_templates': len(self.templates),
            'intent_distribution': intent_counts,
            'complexity_distribution': complexity_counts,
            'average_confidence': avg_confidence,
            'template_usage': template_usage,
            'cache_hit_rate': len(self.generation_cache) / max(1, len(self.generated_prompts))
        }
    
    def export_data(self, output_path: str = None) -> str:
        """导出提示数据"""
        if output_path is None:
            output_path = f"ml_prompts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'templates': {
                tid: asdict(template) for tid, template in self.templates.items()
            },
            'generated_prompts': {
                pid: asdict(prompt) for pid, prompt in self.generated_prompts.items()
            },
            'statistics': self.get_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 提示数据已导出到: {output_path}")
        return output_path

# 测试函数
async def test_intelligent_prompt_generator():
    """测试智能提示生成器"""
    print("🚀 开始智能提示生成器测试...")
    
    # 创建提示生成器
    generator = IntelligentMLPromptGenerator(".")
    
    # 创建测试上下文
    context = PromptGenerationContext(
        user_input="",
        ml_context={
            'available_datasets': [{'name': 'sales_data.csv', 'type': 'structured_text'}],
            'existing_models': [{'name': 'model.pkl', 'framework': 'scikit-learn'}],
            'ml_frameworks': ['scikit-learn', 'pandas', 'matplotlib']
        },
        current_task="数据分析",
        available_tools=['preprocess_data', 'train_model', 'visualize_data'],
        data_context={'current_dataset': 'sales_data.csv'}
    )
    
    # 测试不同类型的提示生成
    test_inputs = [
        "我想分析sales_data.csv数据集，了解销售趋势",
        "帮我训练一个随机森林模型来预测销售额",
        "我的模型准确率只有60%，怎么提升？",
        "创建一个销售数据的可视化图表",
        "代码报错：ValueError: X has 10 features but model expects 8"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n📝 测试 {i}: {user_input}")
        
        context.user_input = user_input
        generated_prompt = await generator.generate_prompt(user_input, context)
        
        print(f"🎯 意图: {generated_prompt.intent.primary_intent.value}")
        print(f"🎲 置信度: {generated_prompt.confidence:.2f}")
        print(f"📋 生成的提示:")
        print(generated_prompt.generated_text)
        print(f"💡 建议: {'; '.join(generated_prompt.suggestions[:2])}")
        print("-" * 50)
    
    # 获取统计信息
    stats = generator.get_statistics()
    print(f"📊 生成统计: {stats}")
    
    # 导出数据
    export_path = generator.export_data()
    print(f"📋 数据导出: {export_path}")

# 兼容原有接口，保持向后兼容

class PromptTemplate(Enum):
    """提示模板类型"""
    DATA_ANALYSIS = "data_analysis"
    MODEL_TRAINING = "model_training"
    VISUALIZATION = "visualization"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    EXPLANATION = "explanation"

class ContextType(Enum):
    """上下文类型"""
    TECHNICAL = "technical"
    BUSINESS = "business"
    EDUCATIONAL = "educational"
    RESEARCH = "research"

@dataclass
class IntelligentPrompt:
    """智能提示数据结构"""
    template_type: PromptTemplate
    context_type: ContextType
    base_prompt: str
    dynamic_sections: Dict[str, str]
    examples: List[str]
    best_practices: List[str]
    common_pitfalls: List[str]
    
    def generate(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """生成完整的智能提示"""
        sections = [
            self._generate_system_identity(),
            self._generate_context_section(context),
            self._generate_capability_section(),
            self._generate_memory_section(context),
            self._generate_task_analysis_section(user_input, context),
            self._generate_best_practices_section(),
            self._generate_response_guidelines(),
            self._generate_user_input_section(user_input)
        ]
        
        return "\n\n".join(filter(None, sections))
    
    def _generate_system_identity(self) -> str:
        """生成系统身份部分"""
        return """# AutoML Workflow Agent - 增强型AI助手

你是一个具备高级机器学习知识和实践经验的AI助手，专门设计用于：
- 🧠 智能分析用户需求并制定最优解决方案
- 🔧 在安全的Docker环境中执行复杂的ML工作流
- 📊 提供数据驱动的洞察和建议
- 🚀 优化模型性能和计算效率
- 📚 传授最佳实践和行业标准

## 核心能力
- **智能推理**: 基于上下文和历史经验进行深度分析
- **记忆系统**: 学习和记住用户偏好与项目特点
- **任务规划**: 自动分解复杂任务为可执行步骤
- **错误预防**: 主动识别潜在问题并提供解决方案"""
    
    def _generate_context_section(self, context: Dict[str, Any]) -> str:
        """生成上下文部分"""
        if not context:
            return ""
        
        sections = ["## 当前上下文"]
        
        if context.get("session_info"):
            sections.append(f"**会话信息**: {context['session_info']}")
        
        if context.get("data_info"):
            sections.append(f"**数据信息**: {context['data_info']}")
        
        if context.get("previous_tasks"):
            sections.append("**历史任务**:")
            for task in context["previous_tasks"]:
                sections.append(f"- {task}")
        
        if context.get("user_preferences"):
            sections.append("**用户偏好**:")
            for pref, value in context["user_preferences"].items():
                sections.append(f"- {pref}: {value}")
        
        return "\n".join(sections)
    
    def _generate_capability_section(self) -> str:
        """生成能力说明部分"""
        return f"""## 可用工具和能力

### 🔧 代码执行环境
- **安全沙箱**: Docker隔离环境，支持GPU加速
- **ML库支持**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **深度学习**: PyTorch, TensorFlow (CPU/GPU)
- **数据处理**: 大规模数据集处理和分析

### 📊 数据分析
- **探索性分析**: 自动生成数据分布、相关性分析
- **质量评估**: 缺失值、异常值、数据偏斜检测
- **可视化**: 交互式图表和统计图形

### 🤖 机器学习
- **自动特征工程**: 特征选择、创建、转换
- **模型选择**: 基于数据特点推荐最适合的算法
- **超参数优化**: 自动调参和交叉验证
- **模型评估**: 全面的性能指标和解释性分析

### 💾 数据管理
- **历史记录**: 完整的实验追踪和版本控制
- **结果保存**: 结构化存储ML成果和洞察
- **数据血缘**: 跟踪数据处理和转换过程"""
    
    def _generate_memory_section(self, context: Dict[str, Any]) -> str:
        """生成记忆部分"""
        if not context or not context.get("relevant_memories"):
            return ""
        
        sections = ["## 📚 相关记忆和经验"]
        
        for memory in context["relevant_memories"]:
            importance_stars = "⭐" * min(5, int(memory.get("importance", 0) * 5))
            sections.append(f"{importance_stars} {memory.get('content', '')}")
        
        return "\n".join(sections)
    
    def _generate_task_analysis_section(self, user_input: str, context: Dict[str, Any]) -> str:
        """生成任务分析部分"""
        sections = ["## 🎯 任务分析框架"]
        
        # 添加基于模板类型的分析框架
        if self.template_type == PromptTemplate.DATA_ANALYSIS:
            sections.extend([
                "### 数据分析检查清单",
                "1. **数据理解**: 形状、类型、分布、质量",
                "2. **业务理解**: 目标、约束、成功指标",
                "3. **探索策略**: 单变量、双变量、多变量分析",
                "4. **洞察提取**: 模式、异常、关系、趋势"
            ])
        elif self.template_type == PromptTemplate.MODEL_TRAINING:
            sections.extend([
                "### 建模流程框架",
                "1. **问题定义**: 监督/无监督、分类/回归/聚类",
                "2. **数据准备**: 清洗、特征工程、划分",
                "3. **模型选择**: 算法比较、复杂度权衡",
                "4. **评估优化**: 指标选择、调参、验证"
            ])
        elif self.template_type == PromptTemplate.VISUALIZATION:
            sections.extend([
                "### 可视化设计原则",
                "1. **目标明确**: 探索性 vs 解释性可视化",
                "2. **图表选择**: 基于数据类型和关系",
                "3. **设计美学**: 色彩、布局、标注",
                "4. **交互性**: 动态筛选、缩放、详情"
            ])
        
        return "\n".join(sections)
    
    def _generate_best_practices_section(self) -> str:
        """生成最佳实践部分"""
        general_practices = [
            "🔍 **数据优先**: 始终从理解数据开始",
            "📐 **渐进式开发**: 从简单模型开始，逐步优化",
            "🔒 **安全第一**: 所有操作在沙箱环境中进行",
            "📊 **可解释性**: 确保结果可以向业务方解释",
            "🧪 **实验追踪**: 记录所有尝试和结果",
            "♻️ **代码复用**: 将有效方法保存为模板"
        ]
        
        template_specific = []
        if self.template_type == PromptTemplate.DATA_ANALYSIS:
            template_specific = [
                "📋 **完整性检查**: 验证数据完整性和一致性",
                "🎯 **目标导向**: 分析要与业务目标对齐",
                "📈 **统计显著性**: 避免偶然模式的过度解读"
            ]
        elif self.template_type == PromptTemplate.MODEL_TRAINING:
            template_specific = [
                "⚖️ **基线对比**: 总是建立简单基线模型",
                "🔄 **交叉验证**: 使用合适的验证策略",
                "🚫 **避免过拟合**: 监控训练和验证性能差异"
            ]
        
        all_practices = general_practices + template_specific
        return "## 💡 最佳实践指南\n\n" + "\n".join(all_practices)
    
    def _generate_response_guidelines(self) -> str:
        """生成响应指导原则"""
        return """## 📝 响应指导原则

### 🎨 响应结构
1. **简明摘要**: 首先提供1-2句话的核心回答
2. **详细分析**: 深入解释方法、原理和考虑因素
3. **具体步骤**: 提供可执行的操作步骤
4. **代码示例**: 给出完整、可运行的代码
5. **期望结果**: 说明预期输出和如何解释

### 🎯 质量标准
- **准确性**: 确保技术内容正确无误
- **完整性**: 覆盖问题的所有重要方面
- **实用性**: 提供立即可用的解决方案
- **教育性**: 解释背后的原理和最佳实践
- **安全性**: 所有操作符合安全规范

### 🚀 增值服务
- **主动建议**: 提供相关的改进建议
- **风险提醒**: 指出潜在的问题和解决方案
- **资源推荐**: 推荐相关工具、库或学习资源
- **后续规划**: 建议下一步的行动方向"""
    
    def _generate_user_input_section(self, user_input: str) -> str:
        """生成用户输入部分"""
        return f"""## 💬 用户请求

{user_input}

---

请基于以上所有信息，提供一个全面、专业且实用的响应。记住要：
1. 充分利用你的专业知识和经验
2. 考虑上下文和历史信息
3. 提供具体可执行的解决方案
4. 主动预防可能的问题
5. 确保响应的教育价值和实用性"""

class IntelligentPromptGenerator:
    """智能提示生成器"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.context_enhancers = self._load_context_enhancers()
        
    def _load_templates(self) -> Dict[PromptTemplate, IntelligentPrompt]:
        """加载提示模板"""
        templates = {}
        
        # 数据分析模板
        templates[PromptTemplate.DATA_ANALYSIS] = IntelligentPrompt(
            template_type=PromptTemplate.DATA_ANALYSIS,
            context_type=ContextType.TECHNICAL,
            base_prompt="数据分析专家提示",
            dynamic_sections={},
            examples=[
                "df.describe() 获取描述性统计",
                "df.info() 查看数据类型和缺失值",
                "sns.pairplot(df) 创建配对图"
            ],
            best_practices=[
                "始终从数据质量评估开始",
                "使用多种可视化方法探索数据",
                "验证假设和发现"
            ],
            common_pitfalls=[
                "忽略缺失值的处理",
                "过度解读相关性",
                "忽略数据分布的偏斜"
            ]
        )
        
        # 模型训练模板
        templates[PromptTemplate.MODEL_TRAINING] = IntelligentPrompt(
            template_type=PromptTemplate.MODEL_TRAINING,
            context_type=ContextType.TECHNICAL,
            base_prompt="机器学习建模专家提示",
            dynamic_sections={},
            examples=[
                "from sklearn.model_selection import train_test_split",
                "from sklearn.ensemble import RandomForestClassifier",
                "from sklearn.metrics import classification_report"
            ],
            best_practices=[
                "建立基线模型进行比较",
                "使用交叉验证评估模型",
                "监控过拟合和欠拟合"
            ],
            common_pitfalls=[
                "数据泄露问题",
                "不平衡数据集处理不当",
                "超参数调优过度"
            ]
        )
        
        # 可视化模板
        templates[PromptTemplate.VISUALIZATION] = IntelligentPrompt(
            template_type=PromptTemplate.VISUALIZATION,
            context_type=ContextType.TECHNICAL,
            base_prompt="数据可视化专家提示",
            dynamic_sections={},
            examples=[
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "plt.figure(figsize=(12, 8))"
            ],
            best_practices=[
                "选择合适的图表类型",
                "注意颜色和标注",
                "确保可读性和美观性"
            ],
            common_pitfalls=[
                "图表过于复杂",
                "颜色选择不当",
                "缺少必要的标注"
            ]
        )
        
        return templates
    
    def _load_context_enhancers(self) -> Dict[str, Any]:
        """加载上下文增强器"""
        return {
            "data_quality_checklist": [
                "检查数据形状和基本信息",
                "识别缺失值模式",
                "检测异常值和离群点",
                "验证数据类型一致性",
                "评估数据分布特征"
            ],
            "ml_workflow_steps": [
                "问题定义和目标设定",
                "数据收集和理解",
                "数据预处理和特征工程",
                "模型选择和训练",
                "模型评估和优化",
                "模型部署和监控"
            ],
            "common_algorithms": {
                "分类": ["RandomForest", "XGBoost", "SVM", "LogisticRegression"],
                "回归": ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor"],
                "聚类": ["KMeans", "DBSCAN", "HierarchicalClustering"],
                "降维": ["PCA", "t-SNE", "UMAP"]
            }
        }
    
    def infer_template_type(self, user_input: str, context: Dict[str, Any] = None) -> PromptTemplate:
        """推断提示模板类型"""
        user_input_lower = user_input.lower()
        
        # 数据分析关键词
        analysis_keywords = ["分析", "探索", "统计", "分布", "相关性", "describe", "info", "explore"]
        if any(keyword in user_input_lower for keyword in analysis_keywords):
            return PromptTemplate.DATA_ANALYSIS
        
        # 模型训练关键词
        training_keywords = ["训练", "模型", "算法", "预测", "分类", "回归", "train", "model", "predict"]
        if any(keyword in user_input_lower for keyword in training_keywords):
            return PromptTemplate.MODEL_TRAINING
        
        # 可视化关键词
        viz_keywords = ["图", "可视化", "画", "chart", "plot", "visualization", "graph"]
        if any(keyword in user_input_lower for keyword in viz_keywords):
            return PromptTemplate.VISUALIZATION
        
        # 调试关键词
        debug_keywords = ["错误", "问题", "调试", "bug", "error", "debug", "fix"]
        if any(keyword in user_input_lower for keyword in debug_keywords):
            return PromptTemplate.DEBUGGING
        
        # 优化关键词
        opt_keywords = ["优化", "改进", "提升", "optimize", "improve", "enhance"]
        if any(keyword in user_input_lower for keyword in opt_keywords):
            return PromptTemplate.OPTIMIZATION
        
        # 解释关键词
        explain_keywords = ["解释", "说明", "原理", "explain", "how", "why", "what"]
        if any(keyword in user_input_lower for keyword in explain_keywords):
            return PromptTemplate.EXPLANATION
        
        # 默认返回数据分析
        return PromptTemplate.DATA_ANALYSIS
    
    def generate_intelligent_prompt(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """生成智能提示"""
        try:
            # 推断模板类型
            template_type = self.infer_template_type(user_input, context)
            
            # 获取对应模板
            template = self.templates.get(template_type)
            if not template:
                # 如果没有找到特定模板，使用数据分析模板作为默认
                template = self.templates[PromptTemplate.DATA_ANALYSIS]
            
            # 增强上下文
            enhanced_context = self._enhance_context(context, template_type, user_input)
            
            # 生成智能提示
            intelligent_prompt = template.generate(user_input, enhanced_context)
            
            logger.info(f"Generated intelligent prompt using template: {template_type.value}")
            return intelligent_prompt
            
        except Exception as e:
            logger.error(f"Error generating intelligent prompt: {str(e)}", exc_info=True)
            # 返回基础提示作为备份
            return self._generate_fallback_prompt(user_input, context)
    
    def _enhance_context(self, context: Dict[str, Any], template_type: PromptTemplate, user_input: str) -> Dict[str, Any]:
        """增强上下文信息"""
        enhanced = context.copy() if context else {}
        
        # 添加模板特定的上下文增强
        if template_type == PromptTemplate.DATA_ANALYSIS:
            enhanced["data_quality_checklist"] = self.context_enhancers["data_quality_checklist"]
        elif template_type == PromptTemplate.MODEL_TRAINING:
            enhanced["ml_workflow"] = self.context_enhancers["ml_workflow_steps"]
            enhanced["algorithm_suggestions"] = self.context_enhancers["common_algorithms"]
        
        # 添加时间上下文
        enhanced["current_time"] = datetime.datetime.now().isoformat()
        
        # 添加用户输入分析
        enhanced["input_analysis"] = {
            "length": len(user_input),
            "complexity": "high" if len(user_input) > 100 else "medium" if len(user_input) > 50 else "low",
            "contains_code": "```" in user_input or "def " in user_input or "import " in user_input
        }
        
        return enhanced
    
    def _generate_fallback_prompt(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """生成备用提示"""
        return f"""# AutoML Workflow Agent

你是一个专业的机器学习助手，请基于以下用户请求提供专业、详细的帮助：

## 用户请求
{user_input}

## 响应要求
1. 提供准确、实用的解决方案
2. 包含具体的代码示例
3. 解释关键概念和最佳实践
4. 考虑安全性和效率
5. 主动提供相关建议

请现在开始响应用户的请求。"""

# 使用示例
if __name__ == "__main__":
    generator = IntelligentPromptGenerator()
    
    # 测试不同类型的用户输入
    test_inputs = [
        "帮我分析这个数据集的质量",
        "训练一个分类模型来预测客户流失",
        "创建一个可视化展示销售趋势",
        "我的模型出现了过拟合问题，怎么解决？"
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"用户输入: {user_input}")
        print(f"{'='*50}")
        
        prompt = generator.generate_intelligent_prompt(
            user_input=user_input,
            context={
                "session_info": "测试会话",
                "relevant_memories": [
                    {"content": "用户偏好使用scikit-learn", "importance": 0.8},
                    {"content": "之前成功处理过客户数据", "importance": 0.6}
                ]
            }
        )
        
        print(prompt[:500] + "...")  # 显示前500个字符