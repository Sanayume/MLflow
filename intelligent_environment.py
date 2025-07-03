# intelligent_environment.py
"""
智能ML环境感知系统
基于gemini-cli的TypeScript架构移植到Python
提供强大的环境分析和上下文理解能力
"""

import asyncio
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
import pkg_resources
import psutil
import pandas as pd
import hashlib
import os
from datetime import datetime

@dataclass
class DatasetInfo:
    path: str
    size_bytes: int
    type: str
    rows: Optional[int] = None
    columns: Optional[int] = None
    schema: Optional[Dict[str, str]] = None
    quality_score: Optional[float] = None
    last_modified: Optional[str] = None

@dataclass
class ModelInfo:
    path: str
    type: str
    framework: str
    size_bytes: int
    created: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ComputeResource:
    type: str
    name: str
    memory_total: int
    memory_available: int
    utilization: float
    temperature: Optional[float] = None

@dataclass
class MLEnvironmentContext:
    """完整的ML环境上下文"""
    # 基础环境信息
    working_dir: str
    platform: str
    python_version: str
    timestamp: str
    
    # 项目结构
    project_type: str
    project_patterns: List[str]
    quality_indicators: Dict[str, bool]
    
    # 数据资源
    available_datasets: List[DatasetInfo]
    data_volume_gb: float
    
    # 计算资源
    compute_resources: List[ComputeResource]
    gpu_available: bool
    
    # 软件环境
    installed_packages: Dict[str, str]
    ml_frameworks: List[str]
    
    # 现有资产
    existing_models: List[ModelInfo]
    notebook_files: List[str]
    script_files: List[str]
    
    # 配置文件
    config_files: List[str]
    environment_files: List[str]

class IntelligentMLEnvironmentAnalyzer:
    """智能ML环境分析器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1小时缓存
        
        # ML相关文件扩展名
        self.dataset_extensions = {
            '.csv', '.parquet', '.json', '.jsonl', '.xlsx', '.xls',
            '.h5', '.hdf5', '.npz', '.npy', '.pkl', '.pickle',
            '.tsv', '.txt', '.data'
        }
        
        self.model_extensions = {
            '.pkl', '.pickle', '.joblib', '.h5', '.hdf5', '.pb',
            '.pth', '.pt', '.onnx', '.tflite', '.model'
        }
        
        self.notebook_extensions = {'.ipynb'}
        self.script_extensions = {'.py', '.R', '.sql'}
        
    async def analyze_environment(self) -> MLEnvironmentContext:
        """主要分析入口：智能分析整个ML环境"""
        
        # 生成缓存键
        cache_key = self._generate_cache_key()
        
        # 检查缓存
        if cache_key in self.analysis_cache:
            cached_result, timestamp = self.analysis_cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                print(f"📊 使用缓存的环境分析结果")
                return cached_result
        
        print(f"🔍 开始智能环境分析: {self.project_root}")
        
        # 并行执行所有分析任务
        tasks = [
            self._analyze_project_structure(),
            self._discover_datasets(),
            self._analyze_compute_resources(),
            self._analyze_software_environment(),
            self._discover_existing_models(),
            self._analyze_code_files(),
            self._discover_config_files()
        ]
        
        try:
            results = await asyncio.gather(*tasks)
            
            # 构建完整的环境上下文
            context = MLEnvironmentContext(
                # 基础信息
                working_dir=str(self.project_root),
                platform=platform.platform(),
                python_version=platform.python_version(),
                timestamp=datetime.now().isoformat(),
                
                # 项目结构分析
                project_type=results[0]['type'],
                project_patterns=results[0]['patterns'],
                quality_indicators=results[0]['quality_indicators'],
                
                # 数据资源
                available_datasets=results[1]['datasets'],
                data_volume_gb=results[1]['total_size_gb'],
                
                # 计算资源
                compute_resources=results[2]['resources'],
                gpu_available=results[2]['gpu_available'],
                
                # 软件环境
                installed_packages=results[3]['packages'],
                ml_frameworks=results[3]['ml_frameworks'],
                
                # 现有资产
                existing_models=results[4]['models'],
                notebook_files=results[5]['notebooks'],
                script_files=results[5]['scripts'],
                
                # 配置
                config_files=results[6]['config_files'],
                environment_files=results[6]['env_files']
            )
            
            # 缓存结果
            self.analysis_cache[cache_key] = (context, datetime.now().timestamp())
            
            print(f"✅ 环境分析完成：发现 {len(context.available_datasets)} 个数据集，{len(context.existing_models)} 个模型")
            return context
            
        except Exception as e:
            print(f"❌ 环境分析失败: {str(e)}")
            raise
    
    async def _analyze_project_structure(self) -> Dict[str, Any]:
        """分析项目结构和类型"""
        structure = {
            'type': 'unknown',
            'patterns': [],
            'quality_indicators': {},
            'framework_indicators': {}
        }
        
        # 检测项目类型特征
        type_indicators = {
            'research': ['notebooks/', 'experiments/', 'research/', '*.ipynb'],
            'production': ['src/', 'app/', 'api/', 'main.py', 'app.py'],
            'ml_pipeline': ['pipelines/', 'dags/', 'workflows/', 'pipeline.py'],
            'data_science': ['data/', 'datasets/', 'models/', 'analysis/'],
            'experiment': ['experiments/', 'trials/', 'runs/', 'mlruns/']
        }
        
        detected_types = []
        for proj_type, indicators in type_indicators.items():
            score = 0
            for indicator in indicators:
                if '*' in indicator:
                    # Glob pattern
                    if list(self.project_root.glob(indicator)):
                        score += 2
                else:
                    # Directory or file
                    if (self.project_root / indicator).exists():
                        score += 1
            
            if score > 0:
                detected_types.append((proj_type, score))
        
        # 选择得分最高的类型
        if detected_types:
            detected_types.sort(key=lambda x: x[1], reverse=True)
            structure['type'] = detected_types[0][0]
            structure['patterns'] = [t[0] for t in detected_types if t[1] > 0]
        
        # 质量指标检测
        structure['quality_indicators'] = {
            'has_tests': any([
                (self.project_root / d).exists() 
                for d in ['tests/', 'test/', '__tests__']
            ]),
            'has_docs': any([
                (self.project_root / d).exists() 
                for d in ['docs/', 'documentation/', 'README.md']
            ]),
            'has_config': any([
                (self.project_root / f).exists() 
                for f in ['config.yml', 'config.yaml', 'config.json', 'settings.py']
            ]),
            'has_requirements': any([
                (self.project_root / f).exists() 
                for f in ['requirements.txt', 'environment.yml', 'pyproject.toml', 'Pipfile']
            ]),
            'has_version_control': (self.project_root / '.git').exists(),
            'has_ci_cd': any([
                (self.project_root / d).exists() 
                for d in ['.github/', '.gitlab-ci.yml', 'Jenkinsfile']
            ]),
            'has_docker': any([
                (self.project_root / f).exists() 
                for f in ['Dockerfile', 'docker-compose.yml']
            ])
        }
        
        return structure
    
    async def _discover_datasets(self) -> Dict[str, Any]:
        """智能发现和分析数据集"""
        datasets = []
        total_size = 0
        
        # 搜索所有可能的数据文件
        for ext in self.dataset_extensions:
            pattern = f"**/*{ext}"
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        dataset_info = await self._analyze_dataset_file(file_path)
                        datasets.append(dataset_info)
                        total_size += dataset_info.size_bytes
                    except Exception as e:
                        print(f"⚠️ 无法分析数据集 {file_path}: {str(e)}")
        
        return {
            'datasets': datasets,
            'total_size_gb': total_size / (1024**3),
            'count': len(datasets)
        }
    
    async def _analyze_dataset_file(self, file_path: Path) -> DatasetInfo:
        """分析单个数据集文件"""
        stat = file_path.stat()
        file_ext = file_path.suffix.lower()
        
        dataset_info = DatasetInfo(
            path=str(file_path.relative_to(self.project_root)),
            size_bytes=stat.st_size,
            type=self._infer_dataset_type(file_ext),
            last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
        )
        
        # 尝试读取数据集元信息（仅对较小的文件）
        if stat.st_size < 100 * 1024 * 1024:  # 小于100MB
            try:
                if file_ext == '.csv':
                    df = pd.read_csv(file_path, nrows=1000)  # 只读前1000行
                    dataset_info.rows = len(df)
                    dataset_info.columns = len(df.columns)
                    dataset_info.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    dataset_info.quality_score = self._calculate_data_quality(df)
                    
                elif file_ext in ['.parquet']:
                    df = pd.read_parquet(file_path)
                    dataset_info.rows = len(df)
                    dataset_info.columns = len(df.columns)
                    dataset_info.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    dataset_info.quality_score = self._calculate_data_quality(df)
                    
            except Exception as e:
                print(f"⚠️ 无法读取数据集详情 {file_path}: {str(e)}")
        
        return dataset_info
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """计算数据质量分数 (0-1)"""
        quality_score = 1.0
        
        # 缺失值惩罚
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality_score -= missing_ratio * 0.3
        
        # 重复值惩罚
        if len(df) > 1:
            duplicate_ratio = df.duplicated().sum() / len(df)
            quality_score -= duplicate_ratio * 0.2
        
        # 数据类型一致性检查
        inconsistent_types = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # 检查数值列是否被错误识别为对象类型
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    inconsistent_types += 1
                except:
                    pass
        
        if len(df.columns) > 0:
            type_inconsistency = inconsistent_types / len(df.columns)
            quality_score -= type_inconsistency * 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _infer_dataset_type(self, file_ext: str) -> str:
        """推断数据集类型"""
        type_mapping = {
            '.csv': 'structured_text',
            '.parquet': 'structured_binary',
            '.json': 'semi_structured',
            '.jsonl': 'semi_structured',
            '.xlsx': 'structured_spreadsheet',
            '.h5': 'scientific_binary',
            '.npz': 'numpy_array',
            '.pkl': 'python_object',
            '.txt': 'unstructured_text'
        }
        return type_mapping.get(file_ext, 'unknown')
    
    async def _analyze_compute_resources(self) -> Dict[str, Any]:
        """分析计算资源"""
        resources = []
        
        # CPU信息
        cpu_info = ComputeResource(
            type='CPU',
            name=platform.processor() or 'Unknown CPU',
            memory_total=psutil.virtual_memory().total,
            memory_available=psutil.virtual_memory().available,
            utilization=psutil.cpu_percent(interval=1)
        )
        resources.append(cpu_info)
        
        # GPU检测
        gpu_available = False
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info = ComputeResource(
                    type='GPU',
                    name=gpu.name,
                    memory_total=int(gpu.memoryTotal * 1024 * 1024),  # MB to bytes
                    memory_available=int(gpu.memoryFree * 1024 * 1024),
                    utilization=gpu.load * 100,
                    temperature=gpu.temperature
                )
                resources.append(gpu_info)
                gpu_available = True
        except ImportError:
            # 尝试nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(', ')
                            if len(parts) >= 4:
                                gpu_info = ComputeResource(
                                    type='GPU',
                                    name=parts[0],
                                    memory_total=int(parts[1]) * 1024 * 1024,  # MB to bytes
                                    memory_available=int(parts[2]) * 1024 * 1024,
                                    utilization=float(parts[3])
                                )
                                resources.append(gpu_info)
                                gpu_available = True
            except Exception:
                pass
        
        return {
            'resources': resources,
            'gpu_available': gpu_available,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    async def _analyze_software_environment(self) -> Dict[str, Any]:
        """分析软件环境"""
        packages = {}
        ml_frameworks = []
        
        # 获取已安装的包
        try:
            installed_packages = [d for d in pkg_resources.working_set]
            for pkg in installed_packages:
                packages[pkg.project_name] = pkg.version
        except Exception as e:
            print(f"⚠️ 无法获取包信息: {str(e)}")
        
        # 检测ML框架
        ml_framework_indicators = {
            'tensorflow': ['tensorflow', 'tensorflow-gpu'],
            'pytorch': ['torch', 'pytorch'],
            'scikit-learn': ['scikit-learn', 'sklearn'],
            'xgboost': ['xgboost'],
            'lightgbm': ['lightgbm'],
            'catboost': ['catboost'],
            'keras': ['keras'],
            'transformers': ['transformers'],
            'opencv': ['opencv-python', 'cv2'],
            'pandas': ['pandas'],
            'numpy': ['numpy'],
            'matplotlib': ['matplotlib'],
            'seaborn': ['seaborn'],
            'plotly': ['plotly'],
            'jupyter': ['jupyter', 'jupyterlab']
        }
        
        for framework, package_names in ml_framework_indicators.items():
            for pkg_name in package_names:
                if pkg_name in packages:
                    ml_frameworks.append(framework)
                    break
        
        return {
            'packages': packages,
            'ml_frameworks': list(set(ml_frameworks)),
            'package_count': len(packages)
        }
    
    async def _discover_existing_models(self) -> Dict[str, Any]:
        """发现现有的模型文件"""
        models = []
        
        for ext in self.model_extensions:
            pattern = f"**/*{ext}"
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        model_info = await self._analyze_model_file(file_path)
                        models.append(model_info)
                    except Exception as e:
                        print(f"⚠️ 无法分析模型文件 {file_path}: {str(e)}")
        
        return {
            'models': models,
            'count': len(models)
        }
    
    async def _analyze_model_file(self, file_path: Path) -> ModelInfo:
        """分析模型文件"""
        stat = file_path.stat()
        file_ext = file_path.suffix.lower()
        
        # 推断框架
        framework = 'unknown'
        if file_ext in ['.pkl', '.pickle', '.joblib']:
            framework = 'scikit-learn/pickle'
        elif file_ext in ['.h5', '.hdf5']:
            framework = 'tensorflow/keras'
        elif file_ext in ['.pth', '.pt']:
            framework = 'pytorch'
        elif file_ext == '.pb':
            framework = 'tensorflow'
        elif file_ext == '.onnx':
            framework = 'onnx'
        
        return ModelInfo(
            path=str(file_path.relative_to(self.project_root)),
            type=self._infer_model_type(file_ext),
            framework=framework,
            size_bytes=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime).isoformat()
        )
    
    def _infer_model_type(self, file_ext: str) -> str:
        """推断模型类型"""
        type_mapping = {
            '.pkl': 'machine_learning_model',
            '.h5': 'deep_learning_model',
            '.pth': 'deep_learning_model',
            '.pb': 'tensorflow_graph',
            '.onnx': 'onnx_model',
            '.tflite': 'mobile_optimized_model'
        }
        return type_mapping.get(file_ext, 'unknown_model')
    
    async def _analyze_code_files(self) -> Dict[str, Any]:
        """分析代码文件"""
        notebooks = []
        scripts = []
        
        # 查找Jupyter notebooks
        for notebook_file in self.project_root.glob("**/*.ipynb"):
            notebooks.append(str(notebook_file.relative_to(self.project_root)))
        
        # 查找Python脚本
        for script_file in self.project_root.glob("**/*.py"):
            scripts.append(str(script_file.relative_to(self.project_root)))
        
        return {
            'notebooks': notebooks,
            'scripts': scripts,
            'notebook_count': len(notebooks),
            'script_count': len(scripts)
        }
    
    async def _discover_config_files(self) -> Dict[str, Any]:
        """发现配置文件"""
        config_patterns = [
            'config.yml', 'config.yaml', 'config.json',
            'settings.py', 'settings.yml', 'settings.yaml',
            '*.conf', '*.ini', '*.toml'
        ]
        
        env_patterns = [
            '.env', '.env.*', 'environment.yml', 
            'requirements.txt', 'pyproject.toml', 'Pipfile'
        ]
        
        config_files = []
        env_files = []
        
        for pattern in config_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    config_files.append(str(file_path.relative_to(self.project_root)))
        
        for pattern in env_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    env_files.append(str(file_path.relative_to(self.project_root)))
        
        return {
            'config_files': config_files,
            'env_files': env_files
        }
    
    def _generate_cache_key(self) -> str:
        """生成分析缓存键"""
        # 基于项目路径和最近修改时间生成键
        key_data = f"{self.project_root}_{os.path.getmtime(self.project_root)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def export_analysis(self, context: MLEnvironmentContext, output_path: str = None) -> str:
        """导出环境分析结果"""
        if output_path is None:
            output_path = f"ml_environment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 转换为可序列化的格式
        data = asdict(context)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 环境分析结果已导出到: {output_path}")
        return output_path
    
    def generate_summary_report(self, context: MLEnvironmentContext) -> str:
        """生成环境分析摘要报告"""
        report = f"""
🤖 ML环境智能分析报告
{'='*50}

📂 项目概览:
  类型: {context.project_type}
  路径: {context.working_dir}
  平台: {context.platform}
  Python版本: {context.python_version}

📊 数据资源:
  数据集数量: {len(context.available_datasets)}
  总数据量: {context.data_volume_gb:.2f} GB
  数据类型: {len(set(d.type for d in context.available_datasets))} 种

🧠 模型资产:
  现有模型: {len(context.existing_models)} 个
  框架分布: {len(set(m.framework for m in context.existing_models))} 种

💻 计算资源:
  CPU核心: {psutil.cpu_count()}
  内存总量: {psutil.virtual_memory().total / (1024**3):.1f} GB
  GPU可用: {'是' if context.gpu_available else '否'}

🔧 软件环境:
  ML框架: {len(context.ml_frameworks)} 种 ({', '.join(context.ml_frameworks[:5])})
  Python包: {len(context.installed_packages)} 个

📝 代码资产:
  Jupyter Notebooks: {len(context.notebook_files)} 个
  Python脚本: {len(context.script_files)} 个

✅ 项目质量:
  测试: {'有' if context.quality_indicators.get('has_tests') else '无'}
  文档: {'有' if context.quality_indicators.get('has_docs') else '无'}
  版本控制: {'有' if context.quality_indicators.get('has_version_control') else '无'}
  CI/CD: {'有' if context.quality_indicators.get('has_ci_cd') else '无'}

🎯 智能建议:
{self._generate_intelligent_recommendations(context)}
"""
        return report
    
    def _generate_intelligent_recommendations(self, context: MLEnvironmentContext) -> str:
        """生成智能建议"""
        recommendations = []
        
        # 数据相关建议
        if len(context.available_datasets) == 0:
            recommendations.append("  • 建议添加数据集到项目中以开始ML工作流")
        elif context.data_volume_gb > 10:
            recommendations.append("  • 数据量较大，建议考虑使用分布式处理框架")
        
        # 计算资源建议
        if not context.gpu_available and 'tensorflow' in context.ml_frameworks:
            recommendations.append("  • 检测到深度学习框架但无GPU，建议配置GPU加速")
        
        # 项目结构建议
        if not context.quality_indicators.get('has_tests'):
            recommendations.append("  • 建议添加单元测试以提高代码质量")
        
        if not context.quality_indicators.get('has_docs'):
            recommendations.append("  • 建议添加项目文档以便团队协作")
        
        # 环境管理建议
        if not any('requirements' in f for f in context.environment_files):
            recommendations.append("  • 建议创建requirements.txt管理依赖")
        
        if len(recommendations) == 0:
            recommendations.append("  • 项目结构良好，环境配置完善！")
        
        return '\n'.join(recommendations)

# 异步测试函数
async def test_environment_analyzer():
    """测试环境分析器"""
    analyzer = IntelligentMLEnvironmentAnalyzer(".")
    
    print("🚀 开始智能环境分析测试...")
    context = await analyzer.analyze_environment()
    
    # 生成报告
    report = analyzer.generate_summary_report(context)
    print(report)
    
    # 导出详细分析
    export_path = analyzer.export_analysis(context)
    print(f"\n📋 详细分析已导出: {export_path}")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_environment_analyzer())