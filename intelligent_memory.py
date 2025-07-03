# intelligent_memory.py
"""
智能ML记忆管理系统
基于gemini-cli的记忆架构，结合向量搜索和知识图谱
提供项目级记忆、上下文检索和智能知识管理
"""

import asyncio
import json
import sqlite3
import hashlib
import pickle
import ast
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

@dataclass
class MemoryNode:
    """记忆节点"""
    id: str
    content: str
    memory_type: str  # conversation, execution, insight, knowledge, pattern
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 重要性评分 0-1
    embedding: Optional[np.ndarray] = None
    
    # 关系信息
    related_nodes: Set[str] = field(default_factory=set)
    parent_nodes: Set[str] = field(default_factory=set)
    child_nodes: Set[str] = field(default_factory=set)
    
    # 访问统计
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

@dataclass
class MLInsight:
    """ML洞察"""
    id: str
    title: str
    description: str
    insight_type: str  # pattern, performance, data_quality, model_behavior
    confidence: float
    supporting_evidence: List[str]
    applicable_contexts: List[str]
    discovered_at: datetime = field(default_factory=datetime.now)

@dataclass
class DataLineage:
    """数据血缘"""
    source_id: str
    target_id: str
    transformation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """搜索结果"""
    node: MemoryNode
    similarity: float
    relevance_explanation: str

class IntelligentMLMemoryManager:
    """智能ML记忆管理器"""
    
    def __init__(self, project_root: str = ".", memory_db_path: str = None):
        self.project_root = Path(project_root).resolve()
        self.memory_db_path = memory_db_path or self.project_root / ".mlagent" / "memory.db"
        
        # 确保目录存在
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 内存中的数据结构
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.knowledge_graph = nx.DiGraph()
        self.data_lineage_graph = nx.DiGraph()
        self.insights: Dict[str, MLInsight] = {}
        
        # 向量化和搜索
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = None
        self.document_ids = []
        
        # 缓存
        self.search_cache = {}
        self.cache_ttl = 3600  # 1小时
        
        # 初始化数据库
        self._init_database()
        
        # 加载现有记忆
        asyncio.create_task(self._load_memories())
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.conn = sqlite3.connect(str(self.memory_db_path), check_same_thread=False)
        
        # 创建表
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                timestamp TEXT,
                importance REAL,
                embedding BLOB,
                related_nodes TEXT,
                parent_nodes TEXT,
                child_nodes TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            );
            
            CREATE TABLE IF NOT EXISTS ml_insights (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                confidence REAL,
                supporting_evidence TEXT,
                applicable_contexts TEXT,
                discovered_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS data_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                transformation TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(source_id, target_id, transformation)
            );
            
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_nodes(memory_type);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_nodes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_importance ON memory_nodes(importance);
            CREATE INDEX IF NOT EXISTS idx_insight_type ON ml_insights(insight_type);
        """)
        
        self.conn.commit()
    
    async def _load_memories(self):
        """从数据库加载记忆"""
        print("🧠 加载现有记忆...")
        
        cursor = self.conn.cursor()
        
        # 加载记忆节点
        cursor.execute("SELECT * FROM memory_nodes")
        for row in cursor.fetchall():
            node = self._row_to_memory_node(row)
            self.memory_nodes[node.id] = node
            
            # 添加到知识图谱
            self.knowledge_graph.add_node(node.id, node=node)
            
            # 添加关系边
            for related_id in node.related_nodes:
                if related_id in self.memory_nodes:
                    self.knowledge_graph.add_edge(node.id, related_id, relation='related')
        
        # 加载洞察
        cursor.execute("SELECT * FROM ml_insights")
        for row in cursor.fetchall():
            insight = self._row_to_insight(row)
            self.insights[insight.id] = insight
        
        # 加载数据血缘
        cursor.execute("SELECT * FROM data_lineage")
        for row in cursor.fetchall():
            lineage = self._row_to_lineage(row)
            self.data_lineage_graph.add_edge(
                lineage.source_id, 
                lineage.target_id, 
                transformation=lineage.transformation,
                metadata=lineage.metadata
            )
        
        print(f"✅ 加载完成: {len(self.memory_nodes)} 个记忆节点, {len(self.insights)} 个洞察")
        
        # 重建向量索引
        if self.memory_nodes:
            await self._rebuild_vector_index()
    
    def _row_to_memory_node(self, row) -> MemoryNode:
        """将数据库行转换为记忆节点"""
        return MemoryNode(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            tags=set(json.loads(row[3]) if row[3] else []),
            metadata=json.loads(row[4]) if row[4] else {},
            timestamp=datetime.fromisoformat(row[5]),
            importance=row[6],
            embedding=pickle.loads(row[7]) if row[7] else None,
            related_nodes=set(json.loads(row[8]) if row[8] else []),
            parent_nodes=set(json.loads(row[9]) if row[9] else []),
            child_nodes=set(json.loads(row[10]) if row[10] else []),
            access_count=row[11],
            last_accessed=datetime.fromisoformat(row[12])
        )
    
    def _row_to_insight(self, row) -> MLInsight:
        """将数据库行转换为洞察"""
        return MLInsight(
            id=row[0],
            title=row[1],
            description=row[2],
            insight_type=row[3],
            confidence=row[4],
            supporting_evidence=json.loads(row[5]) if row[5] else [],
            applicable_contexts=json.loads(row[6]) if row[6] else [],
            discovered_at=datetime.fromisoformat(row[7])
        )
    
    def _row_to_lineage(self, row) -> DataLineage:
        """将数据库行转换为数据血缘"""
        return DataLineage(
            source_id=row[1],
            target_id=row[2],
            transformation=row[3],
            metadata=json.loads(row[4]) if row[4] else {}
        )
    
    async def add_memory(self, 
                        content: str, 
                        memory_type: str,
                        tags: Set[str] = None,
                        metadata: Dict[str, Any] = None,
                        importance: float = 0.5) -> str:
        """添加新记忆"""
        
        # 生成唯一ID
        memory_id = hashlib.md5(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # 创建记忆节点
        node = MemoryNode(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            tags=tags or set(),
            metadata=metadata or {},
            importance=importance
        )
        
        # 生成内容向量
        node.embedding = await self._generate_embedding(content)
        
        # 存储到内存
        self.memory_nodes[memory_id] = node
        
        # 添加到知识图谱
        self.knowledge_graph.add_node(memory_id, node=node)
        
        # 自动发现相关记忆
        related_memories = await self._find_related_memories(node)
        for related_id, similarity in related_memories[:5]:  # 取前5个最相关的
            if similarity > 0.3:  # 相似度阈值
                node.related_nodes.add(related_id)
                self.memory_nodes[related_id].related_nodes.add(memory_id)
                self.knowledge_graph.add_edge(memory_id, related_id, relation='related', weight=similarity)
        
        # 保存到数据库
        await self._save_memory_node(node)
        
        # 重建向量索引
        await self._rebuild_vector_index()
        
        print(f"💭 新增记忆: {memory_type} - {content[:50]}...")
        return memory_id
    
    async def add_conversation_memory(self, 
                                   user_input: str, 
                                   agent_response: str,
                                   context: Dict[str, Any] = None) -> str:
        """添加对话记忆"""
        
        conversation_content = f"User: {user_input}\nAgent: {agent_response}"
        
        metadata = {
            'user_input': user_input,
            'agent_response': agent_response,
            'context': context or {},
            'conversation_length': len(user_input) + len(agent_response)
        }
        
        # 自动提取标签
        tags = self._extract_tags_from_text(conversation_content)
        
        # 计算重要性
        importance = self._calculate_conversation_importance(user_input, agent_response, context)
        
        return await self.add_memory(
            content=conversation_content,
            memory_type='conversation',
            tags=tags,
            metadata=metadata,
            importance=importance
        )
    
    async def add_execution_memory(self, 
                                 tool_name: str,
                                 parameters: Dict[str, Any],
                                 result: Any,
                                 success: bool,
                                 execution_time: float) -> str:
        """添加执行记忆"""
        
        content = f"Tool: {tool_name}\nParameters: {json.dumps(parameters, indent=2)}\nResult: {str(result)[:200]}"
        
        metadata = {
            'tool_name': tool_name,
            'parameters': parameters,
            'result': result,
            'success': success,
            'execution_time': execution_time
        }
        
        tags = {tool_name, 'execution'}
        if success:
            tags.add('success')
        else:
            tags.add('failure')
        
        importance = 0.7 if success else 0.9  # 失败的执行更重要，需要学习
        
        return await self.add_memory(
            content=content,
            memory_type='execution',
            tags=tags,
            metadata=metadata,
            importance=importance
        )
    
    async def add_insight(self, 
                         title: str,
                         description: str,
                         insight_type: str,
                         confidence: float = 0.8,
                         supporting_evidence: List[str] = None,
                         applicable_contexts: List[str] = None) -> str:
        """添加ML洞察"""
        
        insight_id = hashlib.md5(f"{title}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        insight = MLInsight(
            id=insight_id,
            title=title,
            description=description,
            insight_type=insight_type,
            confidence=confidence,
            supporting_evidence=supporting_evidence or [],
            applicable_contexts=applicable_contexts or []
        )
        
        self.insights[insight_id] = insight
        
        # 保存到数据库
        await self._save_insight(insight)
        
        # 创建对应的记忆节点
        memory_content = f"Insight: {title}\n{description}"
        await self.add_memory(
            content=memory_content,
            memory_type='insight',
            tags={'insight', insight_type},
            metadata={'insight_id': insight_id, 'confidence': confidence},
            importance=confidence
        )
        
        print(f"💡 新增洞察: {title}")
        return insight_id
    
    async def smart_search(self, 
                          query: str,
                          memory_types: List[str] = None,
                          tags: Set[str] = None,
                          limit: int = 10,
                          min_similarity: float = 0.1) -> List[SearchResult]:
        """智能搜索记忆"""
        
        # 检查缓存
        cache_key = hashlib.md5(f"{query}_{memory_types}_{tags}_{limit}".encode()).hexdigest()
        if cache_key in self.search_cache:
            cached_result, timestamp = self.search_cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                return cached_result
        
        print(f"🔍 智能搜索: {query}")
        
        results = []
        
        if not self.memory_nodes:
            return results
        
        # 生成查询向量
        query_embedding = await self._generate_embedding(query)
        
        # 计算与所有记忆的相似度
        for node_id, node in self.memory_nodes.items():
            # 类型过滤
            if memory_types and node.memory_type not in memory_types:
                continue
            
            # 标签过滤
            if tags and not tags.intersection(node.tags):
                continue
            
            # 计算相似度
            if node.embedding is not None:
                similarity = self._calculate_similarity(query_embedding, node.embedding)
            else:
                # 回退到文本相似度
                similarity = self._calculate_text_similarity(query, node.content)
            
            if similarity >= min_similarity:
                # 生成相关性解释
                explanation = self._generate_relevance_explanation(query, node, similarity)
                
                results.append(SearchResult(
                    node=node,
                    similarity=similarity,
                    relevance_explanation=explanation
                ))
                
                # 更新访问统计
                node.access_count += 1
                node.last_accessed = datetime.now()
        
        # 按相似度排序
        results.sort(key=lambda x: x.similarity, reverse=True)
        results = results[:limit]
        
        # 缓存结果
        self.search_cache[cache_key] = (results, datetime.now().timestamp())
        
        print(f"📊 找到 {len(results)} 个相关记忆")
        return results
    
    async def get_contextual_memories(self, 
                                    current_context: Dict[str, Any],
                                    limit: int = 5) -> List[MemoryNode]:
        """获取上下文相关的记忆"""
        
        contextual_memories = []
        
        # 提取上下文关键信息
        context_keys = []
        if 'current_task' in current_context:
            context_keys.append(current_context['current_task'])
        if 'data_files' in current_context:
            context_keys.extend(current_context['data_files'])
        if 'model_type' in current_context:
            context_keys.append(current_context['model_type'])
        
        # 为每个上下文关键信息搜索相关记忆
        for key in context_keys:
            search_results = await self.smart_search(str(key), limit=3)
            for result in search_results:
                if result.node not in [m.node for m in contextual_memories]:
                    contextual_memories.append(result)
        
        # 按重要性和相似度排序
        contextual_memories.sort(key=lambda x: x.node.importance * x.similarity, reverse=True)
        
        return [result.node for result in contextual_memories[:limit]]
    
    async def discover_patterns(self) -> List[MLInsight]:
        """自动发现模式和洞察"""
        
        print("🔍 自动发现模式...")
        new_insights = []
        
        # 分析执行模式
        execution_patterns = await self._analyze_execution_patterns()
        new_insights.extend(execution_patterns)
        
        # 分析数据使用模式
        data_patterns = await self._analyze_data_usage_patterns()
        new_insights.extend(data_patterns)
        
        # 分析性能模式
        performance_patterns = await self._analyze_performance_patterns()
        new_insights.extend(performance_patterns)
        
        # 保存新发现的洞察
        for insight in new_insights:
            await self.add_insight(
                title=insight['title'],
                description=insight['description'],
                insight_type=insight['type'],
                confidence=insight['confidence'],
                supporting_evidence=insight['evidence']
            )
        
        print(f"💡 发现 {len(new_insights)} 个新模式")
        return new_insights
    
    async def _analyze_execution_patterns(self) -> List[Dict[str, Any]]:
        """分析执行模式"""
        patterns = []
        
        # 获取所有执行记忆
        execution_memories = [
            node for node in self.memory_nodes.values() 
            if node.memory_type == 'execution'
        ]
        
        if len(execution_memories) < 5:
            return patterns
        
        # 分析工具使用频率
        tool_usage = {}
        for memory in execution_memories:
            tool_name = memory.metadata.get('tool_name')
            if tool_name:
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        # 找出最常用的工具
        if tool_usage:
            most_used_tool = max(tool_usage, key=tool_usage.get)
            usage_count = tool_usage[most_used_tool]
            
            if usage_count >= 3:
                patterns.append({
                    'title': f'高频使用工具: {most_used_tool}',
                    'description': f'工具 {most_used_tool} 被使用了 {usage_count} 次，是最常用的工具',
                    'type': 'tool_usage',
                    'confidence': min(0.9, usage_count / len(execution_memories)),
                    'evidence': [f'工具使用次数: {usage_count}']
                })
        
        # 分析失败模式
        failed_executions = [
            memory for memory in execution_memories 
            if not memory.metadata.get('success', True)
        ]
        
        if len(failed_executions) >= 2:
            failure_rate = len(failed_executions) / len(execution_memories)
            patterns.append({
                'title': '执行失败模式',
                'description': f'发现 {len(failed_executions)} 次执行失败，失败率为 {failure_rate:.1%}',
                'type': 'failure_pattern',
                'confidence': min(0.8, failure_rate * 2),
                'evidence': [f'失败次数: {len(failed_executions)}', f'总执行次数: {len(execution_memories)}']
            })
        
        return patterns
    
    async def _analyze_data_usage_patterns(self) -> List[Dict[str, Any]]:
        """分析数据使用模式"""
        patterns = []
        
        # 分析数据血缘
        if len(self.data_lineage_graph.edges()) >= 3:
            # 找出数据处理链
            longest_chain = []
            for node in self.data_lineage_graph.nodes():
                if self.data_lineage_graph.in_degree(node) == 0:  # 源节点
                    chain = self._find_longest_path_from_node(node)
                    if len(chain) > len(longest_chain):
                        longest_chain = chain
            
            if len(longest_chain) >= 3:
                patterns.append({
                    'title': '复杂数据处理链',
                    'description': f'发现长度为 {len(longest_chain)} 的数据处理链',
                    'type': 'data_pipeline',
                    'confidence': 0.7,
                    'evidence': [f'处理链: {" -> ".join(longest_chain)}']
                })
        
        return patterns
    
    async def _analyze_performance_patterns(self) -> List[Dict[str, Any]]:
        """分析性能模式"""
        patterns = []
        
        # 获取执行时间数据
        execution_times = []
        for memory in self.memory_nodes.values():
            if memory.memory_type == 'execution':
                exec_time = memory.metadata.get('execution_time')
                if exec_time:
                    execution_times.append((memory.metadata.get('tool_name'), exec_time))
        
        if len(execution_times) >= 5:
            # 分析平均执行时间
            avg_time = sum(time for _, time in execution_times) / len(execution_times)
            slow_executions = [item for item in execution_times if item[1] > avg_time * 2]
            
            if slow_executions:
                patterns.append({
                    'title': '性能瓶颈识别',
                    'description': f'发现 {len(slow_executions)} 个执行时间异常的操作',
                    'type': 'performance',
                    'confidence': 0.6,
                    'evidence': [f'平均执行时间: {avg_time:.2f}s', f'异常操作数: {len(slow_executions)}']
                })
        
        return patterns
    
    def _find_longest_path_from_node(self, start_node: str) -> List[str]:
        """从指定节点开始找最长路径"""
        longest_path = [start_node]
        
        def dfs(node, current_path):
            nonlocal longest_path
            if len(current_path) > len(longest_path):
                longest_path = current_path.copy()
            
            for successor in self.data_lineage_graph.successors(node):
                dfs(successor, current_path + [successor])
        
        dfs(start_node, [start_node])
        return longest_path
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本向量"""
        # 简化的TF-IDF向量化（实际应用中可以使用更先进的embedding模型）
        try:
            if not hasattr(self, '_temp_vectorizer'):
                self._temp_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                # 用现有的所有文本来训练向量化器
                all_texts = [node.content for node in self.memory_nodes.values()]
                all_texts.append(text)
                self._temp_vectorizer.fit(all_texts)
            
            vector = self._temp_vectorizer.transform([text]).toarray()[0]
            return vector
        except:
            # 如果向量化失败，返回随机向量
            return np.random.random(100)
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算向量相似度"""
        try:
            return cosine_similarity([vec1], [vec2])[0][0]
        except:
            return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（回退方法）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _extract_tags_from_text(self, text: str) -> Set[str]:
        """从文本中提取标签"""
        tags = set()
        
        # ML相关关键词
        ml_keywords = {
            'train', 'training', 'model', 'dataset', 'data', 'predict', 'prediction',
            'accuracy', 'loss', 'validation', 'test', 'feature', 'engineering',
            'preprocessing', 'visualization', 'analysis', 'evaluation', 'metrics'
        }
        
        # 数据科学工具
        tools = {
            'pandas', 'numpy', 'sklearn', 'tensorflow', 'pytorch', 'matplotlib',
            'seaborn', 'plotly', 'xgboost', 'lightgbm', 'catboost'
        }
        
        text_lower = text.lower()
        
        # 提取ML关键词
        for keyword in ml_keywords:
            if keyword in text_lower:
                tags.add(keyword)
        
        # 提取工具名
        for tool in tools:
            if tool in text_lower:
                tags.add(tool)
        
        # 提取文件扩展名相关的标签
        file_patterns = re.findall(r'\.(\w+)', text)
        for ext in file_patterns:
            if ext in ['csv', 'json', 'pkl', 'h5', 'parquet']:
                tags.add(f'data_{ext}')
            elif ext in ['py', 'ipynb', 'R']:
                tags.add(f'code_{ext}')
        
        return tags
    
    def _calculate_conversation_importance(self, 
                                         user_input: str, 
                                         agent_response: str,
                                         context: Dict[str, Any] = None) -> float:
        """计算对话重要性"""
        importance = 0.5
        
        # 长度因子
        total_length = len(user_input) + len(agent_response)
        if total_length > 500:
            importance += 0.1
        
        # 关键词因子
        important_keywords = {
            'error', 'fail', 'problem', 'issue', 'bug', 'fix',
            'important', 'critical', 'urgent', 'remember',
            'model', 'train', 'deploy', 'production'
        }
        
        text = (user_input + ' ' + agent_response).lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        importance += keyword_count * 0.05
        
        # 上下文因子
        if context:
            if context.get('error_occurred'):
                importance += 0.2
            if context.get('model_training'):
                importance += 0.15
        
        return min(1.0, importance)
    
    def _generate_relevance_explanation(self, 
                                      query: str, 
                                      node: MemoryNode, 
                                      similarity: float) -> str:
        """生成相关性解释"""
        explanations = []
        
        # 相似度解释
        if similarity > 0.7:
            explanations.append("高度相关")
        elif similarity > 0.4:
            explanations.append("中等相关")
        else:
            explanations.append("低度相关")
        
        # 类型匹配
        explanations.append(f"记忆类型: {node.memory_type}")
        
        # 标签匹配
        query_words = set(query.lower().split())
        matching_tags = node.tags.intersection(query_words)
        if matching_tags:
            explanations.append(f"匹配标签: {', '.join(matching_tags)}")
        
        # 重要性
        if node.importance > 0.7:
            explanations.append("高重要性")
        
        return " | ".join(explanations)
    
    async def _rebuild_vector_index(self):
        """重建向量索引"""
        if not self.memory_nodes:
            return
        
        print("🔄 重建向量索引...")
        
        # 收集所有文本
        texts = []
        self.document_ids = []
        
        for node_id, node in self.memory_nodes.items():
            texts.append(node.content)
            self.document_ids.append(node_id)
        
        # 训练向量化器
        try:
            self.document_vectors = self.vectorizer.fit_transform(texts)
            print(f"✅ 向量索引重建完成: {len(texts)} 个文档")
        except Exception as e:
            print(f"⚠️ 向量索引重建失败: {str(e)}")
    
    async def _save_memory_node(self, node: MemoryNode):
        """保存记忆节点到数据库"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memory_nodes 
            (id, content, memory_type, tags, metadata, timestamp, importance, 
             embedding, related_nodes, parent_nodes, child_nodes, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            node.content,
            node.memory_type,
            json.dumps(list(node.tags)),
            json.dumps(node.metadata, default=str),
            node.timestamp.isoformat(),
            node.importance,
            pickle.dumps(node.embedding) if node.embedding is not None else None,
            json.dumps(list(node.related_nodes)),
            json.dumps(list(node.parent_nodes)),
            json.dumps(list(node.child_nodes)),
            node.access_count,
            node.last_accessed.isoformat()
        ))
        
        self.conn.commit()
    
    async def _save_insight(self, insight: MLInsight):
        """保存洞察到数据库"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO ml_insights 
            (id, title, description, insight_type, confidence, supporting_evidence, applicable_contexts, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.id,
            insight.title,
            insight.description,
            insight.insight_type,
            insight.confidence,
            json.dumps(insight.supporting_evidence),
            json.dumps(insight.applicable_contexts),
            insight.discovered_at.isoformat()
        ))
        
        self.conn.commit()
    
    async def _find_related_memories(self, node: MemoryNode) -> List[Tuple[str, float]]:
        """找到相关记忆"""
        related = []
        
        if not node.embedding is None:
            for other_id, other_node in self.memory_nodes.items():
                if other_id != node.id and other_node.embedding is not None:
                    similarity = self._calculate_similarity(node.embedding, other_node.embedding)
                    related.append((other_id, similarity))
        
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if not self.memory_nodes:
            return {'total_memories': 0}
        
        # 按类型统计
        type_counts = {}
        for node in self.memory_nodes.values():
            type_counts[node.memory_type] = type_counts.get(node.memory_type, 0) + 1
        
        # 按重要性统计
        importance_levels = {'low': 0, 'medium': 0, 'high': 0}
        for node in self.memory_nodes.values():
            if node.importance < 0.3:
                importance_levels['low'] += 1
            elif node.importance < 0.7:
                importance_levels['medium'] += 1
            else:
                importance_levels['high'] += 1
        
        # 计算平均访问次数
        avg_access = sum(node.access_count for node in self.memory_nodes.values()) / len(self.memory_nodes)
        
        return {
            'total_memories': len(self.memory_nodes),
            'total_insights': len(self.insights),
            'memory_types': type_counts,
            'importance_distribution': importance_levels,
            'knowledge_graph_nodes': len(self.knowledge_graph.nodes()),
            'knowledge_graph_edges': len(self.knowledge_graph.edges()),
            'data_lineage_nodes': len(self.data_lineage_graph.nodes()),
            'data_lineage_edges': len(self.data_lineage_graph.edges()),
            'average_access_count': avg_access
        }
    
    def export_memories(self, output_path: str = None) -> str:
        """导出记忆数据"""
        if output_path is None:
            output_path = f"ml_memories_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'memories': {
                node_id: {
                    'content': node.content,
                    'memory_type': node.memory_type,
                    'tags': list(node.tags),
                    'metadata': node.metadata,
                    'timestamp': node.timestamp.isoformat(),
                    'importance': node.importance,
                    'access_count': node.access_count
                }
                for node_id, node in self.memory_nodes.items()
            },
            'insights': {
                insight_id: asdict(insight)
                for insight_id, insight in self.insights.items()
            },
            'statistics': self.get_memory_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 记忆数据已导出到: {output_path}")
        return output_path

# 测试函数
async def test_intelligent_memory():
    """测试智能记忆管理器"""
    print("🚀 开始智能记忆管理器测试...")
    
    # 创建记忆管理器
    memory_manager = IntelligentMLMemoryManager(".")
    
    # 添加一些测试记忆
    await memory_manager.add_conversation_memory(
        user_input="我想训练一个图像分类模型",
        agent_response="好的，我可以帮你训练图像分类模型。首先需要准备数据集，然后选择合适的神经网络架构。",
        context={'task_type': 'image_classification', 'user_level': 'beginner'}
    )
    
    await memory_manager.add_execution_memory(
        tool_name="preprocess_data",
        parameters={"dataset_path": "images/", "target_size": [224, 224]},
        result={"processed_images": 1000, "validation_split": 0.2},
        success=True,
        execution_time=45.2
    )
    
    await memory_manager.add_execution_memory(
        tool_name="train_model",
        parameters={"model_type": "CNN", "epochs": 50, "batch_size": 32},
        result={"final_accuracy": 0.89, "loss": 0.25},
        success=True,
        execution_time=1800.5
    )
    
    # 搜索测试
    search_results = await memory_manager.smart_search("图像分类模型训练", limit=5)
    print(f"🔍 搜索结果: {len(search_results)} 个相关记忆")
    
    for result in search_results:
        print(f"  - 相似度: {result.similarity:.3f} | {result.relevance_explanation}")
        print(f"    内容: {result.node.content[:100]}...")
    
    # 发现模式
    patterns = await memory_manager.discover_patterns()
    print(f"💡 发现 {len(patterns)} 个模式")
    
    # 获取统计信息
    stats = memory_manager.get_memory_statistics()
    print(f"📊 记忆统计: {stats}")
    
    # 导出记忆
    export_path = memory_manager.export_memories()
    print(f"📋 记忆导出: {export_path}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_memory())