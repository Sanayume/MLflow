"""
Intelligent Prompt Generation System
智能提示生成系统
"""

import json
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

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