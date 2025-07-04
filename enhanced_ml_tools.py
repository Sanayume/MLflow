"""
Enhanced ML Tools and Visualization Module
增强的机器学习工具和可视化模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, silhouette_score
)

logger = logging.getLogger(__name__)

class MLTaskType(Enum):
    """机器学习任务类型"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"

class VisualizationType(Enum):
    """可视化类型"""
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    COMPARISON = "comparison"
    TREND = "trend"
    GEOGRAPHICAL = "geographical"
    NETWORK = "network"
    CUSTOM = "custom"

@dataclass
class MLExperiment:
    """机器学习实验数据结构"""
    experiment_id: str
    task_type: MLTaskType
    algorithm: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    model_size: int
    features_used: List[str]
    target_variable: str
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]
    created_at: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'experiment_id': self.experiment_id,
            'task_type': self.task_type.value,
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'training_time': self.training_time,
            'model_size': self.model_size,
            'features_used': self.features_used,
            'target_variable': self.target_variable,
            'cross_validation_scores': self.cross_validation_scores,
            'feature_importance': self.feature_importance,
            'created_at': self.created_at.isoformat()
        }
        return result

class EnhancedMLTools:
    """增强的机器学习工具"""
    
    def __init__(self, output_dir: str = "agent_workspace/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[MLExperiment] = []
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("Enhanced ML Tools initialized")
    
    def auto_data_analysis(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """自动数据分析"""
        try:
            analysis = {
                'basic_info': self._get_basic_info(df),
                'missing_values': self._analyze_missing_values(df),
                'data_types': self._analyze_data_types(df),
                'numerical_summary': self._get_numerical_summary(df),
                'categorical_summary': self._get_categorical_summary(df),
                'correlations': self._analyze_correlations(df),
                'outliers': self._detect_outliers(df),
                'data_quality_score': self._calculate_data_quality_score(df)
            }
            
            if target_column and target_column in df.columns:
                analysis['target_analysis'] = self._analyze_target_variable(df, target_column)
                analysis['feature_target_relationships'] = self._analyze_feature_target_relationships(df, target_column)
            
            logger.info(f"Completed auto data analysis for dataset with shape {df.shape}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in auto data analysis: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取基本信息"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.to_dict()
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析缺失值"""
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        return {
            'total_missing': missing_count.sum(),
            'missing_by_column': missing_count.to_dict(),
            'missing_percentage': missing_percent.to_dict(),
            'columns_with_missing': missing_count[missing_count > 0].index.tolist(),
            'missing_patterns': self._identify_missing_patterns(df)
        }
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """识别缺失值模式"""
        patterns = []
        
        # 完全缺失的行
        completely_missing_rows = df.isnull().all(axis=1).sum()
        if completely_missing_rows > 0:
            patterns.append({
                'type': 'completely_missing_rows',
                'count': completely_missing_rows,
                'description': '完全缺失的行'
            })
        
        # 高缺失率的列
        high_missing_cols = df.isnull().sum()[df.isnull().sum() / len(df) > 0.5]
        if len(high_missing_cols) > 0:
            patterns.append({
                'type': 'high_missing_columns',
                'columns': high_missing_cols.index.tolist(),
                'description': '缺失率超过50%的列'
            })
        
        return patterns
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析数据类型"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols),
            'datetime_count': len(datetime_cols)
        }
    
    def _get_numerical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数值列摘要"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        
        return {
            'descriptive_stats': numeric_df.describe().to_dict(),
            'skewness': numeric_df.skew().to_dict(),
            'kurtosis': numeric_df.kurtosis().to_dict()
        }
    
    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取分类列摘要"""
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if categorical_df.empty:
            return {}
        
        summary = {}
        for col in categorical_df.columns:
            value_counts = df[col].value_counts()
            summary[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
                'top_5_values': value_counts.head().to_dict()
            }
        
        return summary
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析相关性"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {}
        
        corr_matrix = numeric_df.corr()
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'max_correlation': corr_matrix.abs().max().max(),
            'avg_correlation': corr_matrix.abs().mean().mean()
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测异常值"""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers_summary = {}
        
        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
            
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(numeric_df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outliers_summary
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """计算数据质量分数"""
        factors = []
        
        # 缺失值因子 (0-1, 1最好)
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        missing_factor = 1 - missing_ratio
        factors.append(missing_factor * 0.3)
        
        # 数据类型一致性因子
        type_consistency = 1.0  # 简化实现
        factors.append(type_consistency * 0.2)
        
        # 重复值因子
        duplicate_ratio = df.duplicated().sum() / len(df)
        duplicate_factor = 1 - duplicate_ratio
        factors.append(duplicate_factor * 0.2)
        
        # 数值范围合理性因子
        range_factor = 1.0  # 简化实现
        factors.append(range_factor * 0.3)
        
        return sum(factors)
    
    def _analyze_target_variable(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """分析目标变量"""
        target = df[target_column]
        
        analysis = {
            'data_type': str(target.dtype),
            'unique_count': target.nunique(),
            'missing_count': target.isnull().sum(),
            'missing_percentage': target.isnull().sum() / len(target) * 100
        }
        
        if target.dtype in ['object', 'category']:
            # 分类目标变量
            value_counts = target.value_counts()
            analysis.update({
                'type': 'categorical',
                'class_distribution': value_counts.to_dict(),
                'class_balance': value_counts.min() / value_counts.max(),
                'most_frequent_class': value_counts.index[0],
                'least_frequent_class': value_counts.index[-1]
            })
        else:
            # 数值目标变量
            analysis.update({
                'type': 'numerical',
                'statistics': target.describe().to_dict(),
                'skewness': target.skew(),
                'kurtosis': target.kurtosis()
            })
        
        return analysis
    
    def _analyze_feature_target_relationships(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """分析特征与目标变量的关系"""
        relationships = {}
        target = df[target_column]
        
        # 数值特征与目标的关系
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != target_column]
        
        for feature in numeric_features:
            if target.dtype in ['object', 'category']:
                # 数值特征 vs 分类目标：使用ANOVA F-统计量
                try:
                    from scipy.stats import f_oneway
                    groups = [df[df[target_column] == class_val][feature].dropna() 
                             for class_val in target.unique()]
                    f_stat, p_value = f_oneway(*groups)
                    relationships[feature] = {
                        'type': 'numerical_vs_categorical',
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significance': 'significant' if p_value < 0.05 else 'not_significant'
                    }
                except:
                    relationships[feature] = {'type': 'numerical_vs_categorical', 'error': 'calculation_failed'}
            else:
                # 数值特征 vs 数值目标：使用相关系数
                correlation = df[feature].corr(target)
                relationships[feature] = {
                    'type': 'numerical_vs_numerical',
                    'correlation': correlation,
                    'correlation_strength': self._interpret_correlation(abs(correlation))
                }
        
        return relationships
    
    def _interpret_correlation(self, abs_corr: float) -> str:
        """解释相关性强度"""
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        elif abs_corr >= 0.1:
            return "weak"
        else:
            return "very_weak"
    
    def smart_preprocessing(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """智能预处理"""
        try:
            processed_df = df.copy()
            preprocessing_steps = []
            
            # 1. 处理缺失值
            missing_strategy = self._determine_missing_strategy(df)
            processed_df, missing_steps = self._handle_missing_values(processed_df, missing_strategy)
            preprocessing_steps.extend(missing_steps)
            
            # 2. 处理异常值
            processed_df, outlier_steps = self._handle_outliers(processed_df)
            preprocessing_steps.extend(outlier_steps)
            
            # 3. 编码分类变量
            processed_df, encoding_steps = self._encode_categorical_variables(processed_df, target_column)
            preprocessing_steps.extend(encoding_steps)
            
            # 4. 特征缩放
            processed_df, scaling_steps = self._scale_features(processed_df, target_column)
            preprocessing_steps.extend(scaling_steps)
            
            # 5. 特征选择
            if target_column:
                processed_df, selection_steps = self._select_features(processed_df, target_column)
                preprocessing_steps.extend(selection_steps)
            
            preprocessing_report = {
                'original_shape': df.shape,
                'processed_shape': processed_df.shape,
                'steps_applied': preprocessing_steps,
                'features_removed': list(set(df.columns) - set(processed_df.columns)),
                'features_added': list(set(processed_df.columns) - set(df.columns))
            }
            
            logger.info(f"Smart preprocessing completed. Shape changed from {df.shape} to {processed_df.shape}")
            return processed_df, preprocessing_report
            
        except Exception as e:
            logger.error(f"Error in smart preprocessing: {str(e)}", exc_info=True)
            return df, {"error": str(e)}
    
    def _determine_missing_strategy(self, df: pd.DataFrame) -> Dict[str, str]:
        """确定缺失值处理策略"""
        strategies = {}
        
        for column in df.columns:
            missing_pct = df[column].isnull().sum() / len(df)
            
            if missing_pct == 0:
                strategies[column] = 'none'
            elif missing_pct > 0.5:
                strategies[column] = 'drop'
            elif df[column].dtype in ['object', 'category']:
                strategies[column] = 'mode'
            else:
                strategies[column] = 'median'
        
        return strategies
    
    def _handle_missing_values(self, df: pd.DataFrame, strategies: Dict[str, str]) -> Tuple[pd.DataFrame, List[Dict]]:
        """处理缺失值"""
        processed_df = df.copy()
        steps = []
        
        # 首先删除需要删除的列
        cols_to_drop = [col for col, strategy in strategies.items() if strategy == 'drop']
        if cols_to_drop:
            processed_df = processed_df.drop(columns=cols_to_drop)
            steps.append({
                'step': 'drop_columns',
                'columns': cols_to_drop,
                'reason': 'high_missing_rate'
            })
        
        # 处理其他列的缺失值
        for column, strategy in strategies.items():
            if column not in processed_df.columns:
                continue
                
            if strategy == 'mode':
                mode_value = processed_df[column].mode().iloc[0] if not processed_df[column].mode().empty else 'unknown'
                processed_df[column].fillna(mode_value, inplace=True)
                steps.append({
                    'step': 'fill_mode',
                    'column': column,
                    'fill_value': mode_value
                })
            elif strategy == 'median':
                median_value = processed_df[column].median()
                processed_df[column].fillna(median_value, inplace=True)
                steps.append({
                    'step': 'fill_median',
                    'column': column,
                    'fill_value': median_value
                })
        
        return processed_df, steps
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """处理异常值"""
        processed_df = df.copy()
        steps = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            q1 = processed_df[column].quantile(0.25)
            q3 = processed_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_count = len(processed_df[(processed_df[column] < lower_bound) | (processed_df[column] > upper_bound)])
            
            if outlier_count > 0 and outlier_count / len(processed_df) < 0.05:  # 少于5%的异常值才处理
                # 使用截断方法处理异常值
                processed_df[column] = processed_df[column].clip(lower=lower_bound, upper=upper_bound)
                steps.append({
                    'step': 'clip_outliers',
                    'column': column,
                    'outlier_count': outlier_count,
                    'bounds': [lower_bound, upper_bound]
                })
        
        return processed_df, steps
    
    def _encode_categorical_variables(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """编码分类变量"""
        processed_df = df.copy()
        steps = []
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if target_column:
            categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for column in categorical_columns:
            unique_count = processed_df[column].nunique()
            
            if unique_count <= 10:  # 低基数：使用One-Hot编码
                dummies = pd.get_dummies(processed_df[column], prefix=column)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df.drop(column, axis=1, inplace=True)
                steps.append({
                    'step': 'one_hot_encoding',
                    'column': column,
                    'new_columns': list(dummies.columns)
                })
            else:  # 高基数：使用Label编码
                le = LabelEncoder()
                processed_df[column] = le.fit_transform(processed_df[column].astype(str))
                steps.append({
                    'step': 'label_encoding',
                    'column': column,
                    'classes': list(le.classes_)
                })
        
        return processed_df, steps
    
    def _scale_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """特征缩放"""
        processed_df = df.copy()
        steps = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_columns:
            numeric_columns = [col for col in numeric_columns if col != target_column]
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            processed_df[numeric_columns] = scaler.fit_transform(processed_df[numeric_columns])
            steps.append({
                'step': 'standard_scaling',
                'columns': list(numeric_columns),
                'scaler_params': {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            })
        
        return processed_df, steps
    
    def _select_features(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """特征选择"""
        if target_column not in df.columns:
            return df, []
        
        processed_df = df.copy()
        steps = []
        
        # 移除高相关性的特征
        feature_columns = [col for col in df.columns if col != target_column]
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            
            if high_corr_features:
                processed_df.drop(high_corr_features, axis=1, inplace=True)
                steps.append({
                    'step': 'remove_high_correlation',
                    'removed_features': high_corr_features,
                    'threshold': 0.95
                })
        
        return processed_df, steps
    
    def auto_model_selection(self, X: pd.DataFrame, y: pd.Series, task_type: MLTaskType = None) -> Dict[str, Any]:
        """自动模型选择"""
        try:
            if task_type is None:
                task_type = self._infer_task_type(y)
            
            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if task_type == MLTaskType.CLASSIFICATION else None
            )
            
            # 获取候选模型
            models = self._get_candidate_models(task_type)
            
            # 评估模型
            results = []
            for name, model in models.items():
                try:
                    start_time = datetime.datetime.now()
                    
                    # 交叉验证
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                              scoring=self._get_scoring_metric(task_type))
                    
                    # 训练并评估
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    training_time = (datetime.datetime.now() - start_time).total_seconds()
                    
                    # 计算指标
                    metrics = self._calculate_metrics(y_test, y_pred, task_type)
                    
                    # 特征重要性（如果支持）
                    feature_importance = {}
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        if len(model.coef_.shape) == 1:
                            feature_importance = dict(zip(X.columns, abs(model.coef_)))
                        else:
                            feature_importance = dict(zip(X.columns, abs(model.coef_[0])))
                    
                    experiment = MLExperiment(
                        experiment_id=f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        task_type=task_type,
                        algorithm=name,
                        parameters=model.get_params(),
                        metrics=metrics,
                        training_time=training_time,
                        model_size=self._estimate_model_size(model),
                        features_used=list(X.columns),
                        target_variable=y.name if hasattr(y, 'name') else 'target',
                        cross_validation_scores=cv_scores.tolist(),
                        feature_importance=feature_importance,
                        created_at=datetime.datetime.now()
                    )
                    
                    results.append(experiment)
                    self.experiments.append(experiment)
                    
                except Exception as model_error:
                    logger.warning(f"Error training model {name}: {str(model_error)}")
                    continue
            
            # 排序结果
            if task_type == MLTaskType.CLASSIFICATION:
                results.sort(key=lambda x: x.metrics.get('accuracy', 0), reverse=True)
            else:
                results.sort(key=lambda x: x.metrics.get('r2_score', -float('inf')), reverse=True)
            
            return {
                'task_type': task_type.value,
                'best_model': results[0].to_dict() if results else None,
                'all_results': [exp.to_dict() for exp in results],
                'recommendations': self._generate_model_recommendations(results, task_type)
            }
            
        except Exception as e:
            logger.error(f"Error in auto model selection: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _infer_task_type(self, y: pd.Series) -> MLTaskType:
        """推断任务类型"""
        if y.dtype == 'object' or y.nunique() < 20:
            return MLTaskType.CLASSIFICATION
        else:
            return MLTaskType.REGRESSION
    
    def _get_candidate_models(self, task_type: MLTaskType) -> Dict[str, Any]:
        """获取候选模型"""
        if task_type == MLTaskType.CLASSIFICATION:
            return {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True)
            }
        elif task_type == MLTaskType.REGRESSION:
            return {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'SVR': SVR()
            }
        elif task_type == MLTaskType.CLUSTERING:
            return {
                'KMeans': KMeans(n_clusters=3, random_state=42),
                'DBSCAN': DBSCAN()
            }
        else:
            return {}
    
    def _get_scoring_metric(self, task_type: MLTaskType) -> str:
        """获取评分指标"""
        if task_type == MLTaskType.CLASSIFICATION:
            return 'accuracy'
        elif task_type == MLTaskType.REGRESSION:
            return 'r2'
        else:
            return 'adjusted_rand_score'
    
    def _calculate_metrics(self, y_true, y_pred, task_type: MLTaskType) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        if task_type == MLTaskType.CLASSIFICATION:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
            except:
                metrics['roc_auc'] = 0.0
                
        elif task_type == MLTaskType.REGRESSION:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def _estimate_model_size(self, model) -> int:
        """估算模型大小"""
        try:
            import pickle
            return len(pickle.dumps(model))
        except:
            return 0
    
    def _generate_model_recommendations(self, results: List[MLExperiment], task_type: MLTaskType) -> List[str]:
        """生成模型推荐"""
        recommendations = []
        
        if not results:
            return ["没有成功训练的模型"]
        
        best_model = results[0]
        recommendations.append(f"推荐使用 {best_model.algorithm}，它在这个数据集上表现最好")
        
        if task_type == MLTaskType.CLASSIFICATION:
            if best_model.metrics.get('accuracy', 0) < 0.7:
                recommendations.append("模型准确率较低，建议进一步的特征工程或数据增强")
            
            if len(results) > 1:
                second_best = results[1]
                acc_diff = best_model.metrics.get('accuracy', 0) - second_best.metrics.get('accuracy', 0)
                if acc_diff < 0.02:
                    recommendations.append(f"{second_best.algorithm} 的性能与最佳模型接近，可以考虑模型集成")
        
        elif task_type == MLTaskType.REGRESSION:
            if best_model.metrics.get('r2_score', 0) < 0.5:
                recommendations.append("模型R²分数较低，建议检查特征选择和数据质量")
        
        # 训练时间建议
        if best_model.training_time > 60:
            recommendations.append("最佳模型训练时间较长，生产环境中请考虑性能优化")
        
        return recommendations
    
    def create_advanced_visualizations(self, df: pd.DataFrame, viz_type: VisualizationType, 
                                     **kwargs) -> Dict[str, Any]:
        """创建高级可视化"""
        try:
            if viz_type == VisualizationType.DISTRIBUTION:
                return self._create_distribution_plots(df, **kwargs)
            elif viz_type == VisualizationType.CORRELATION:
                return self._create_correlation_plots(df, **kwargs)
            elif viz_type == VisualizationType.COMPARISON:
                return self._create_comparison_plots(df, **kwargs)
            elif viz_type == VisualizationType.TREND:
                return self._create_trend_plots(df, **kwargs)
            else:
                return {"error": f"Unsupported visualization type: {viz_type}"}
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _create_distribution_plots(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """创建分布图"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns[:6]  # 最多6列
        
        if len(numeric_columns) == 0:
            return {"error": "No numeric columns found for distribution plots"}
        
        # 创建子图
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=list(numeric_columns),
            specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
        )
        
        for i, col in enumerate(numeric_columns):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            # 直方图
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=30),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title="数据分布图",
            height=300 * n_rows,
            showlegend=False
        )
        
        # 保存图片
        plot_path = self.output_dir / f"distribution_plots_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        
        return {
            "plot_path": str(plot_path),
            "plot_type": "distribution",
            "columns_plotted": list(numeric_columns),
            "insights": self._generate_distribution_insights(df, numeric_columns)
        }
    
    def _create_correlation_plots(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """创建相关性图"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns for correlation plot"}
        
        corr_matrix = numeric_df.corr()
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="特征相关性热力图",
            xaxis_title="特征",
            yaxis_title="特征",
            width=800,
            height=800
        )
        
        # 保存图片
        plot_path = self.output_dir / f"correlation_heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        
        return {
            "plot_path": str(plot_path),
            "plot_type": "correlation",
            "correlation_matrix": corr_matrix.to_dict(),
            "insights": self._generate_correlation_insights(corr_matrix)
        }
    
    def _create_comparison_plots(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """创建对比图"""
        group_by = kwargs.get('group_by')
        value_col = kwargs.get('value_col')
        
        if not group_by or not value_col:
            return {"error": "group_by and value_col parameters are required for comparison plots"}
        
        if group_by not in df.columns or value_col not in df.columns:
            return {"error": "Specified columns not found in dataframe"}
        
        # 按组统计
        grouped_stats = df.groupby(group_by)[value_col].agg(['mean', 'std', 'count']).reset_index()
        
        # 创建柱状图
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=grouped_stats[group_by],
            y=grouped_stats['mean'],
            error_y=dict(type='data', array=grouped_stats['std']),
            name='平均值',
            text=grouped_stats['count'],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f"{value_col} 按 {group_by} 分组对比",
            xaxis_title=group_by,
            yaxis_title=f"{value_col} 平均值",
            showlegend=True
        )
        
        # 保存图片
        plot_path = self.output_dir / f"comparison_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        
        return {
            "plot_path": str(plot_path),
            "plot_type": "comparison",
            "group_stats": grouped_stats.to_dict('records'),
            "insights": self._generate_comparison_insights(grouped_stats, group_by, value_col)
        }
    
    def _create_trend_plots(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """创建趋势图"""
        date_col = kwargs.get('date_col')
        value_cols = kwargs.get('value_cols', [])
        
        if not date_col:
            # 尝试自动识别日期列
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0:
                date_col = date_columns[0]
            else:
                return {"error": "No date column specified or found"}
        
        if not value_cols:
            # 使用数值列
            value_cols = df.select_dtypes(include=[np.number]).columns[:3]  # 最多3列
        
        if len(value_cols) == 0:
            return {"error": "No numeric columns found for trend plot"}
        
        # 创建趋势图
        fig = go.Figure()
        
        for col in value_cols:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="时间趋势图",
            xaxis_title=date_col,
            yaxis_title="数值",
            hovermode='x unified',
            showlegend=True
        )
        
        # 保存图片
        plot_path = self.output_dir / f"trend_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        
        return {
            "plot_path": str(plot_path),
            "plot_type": "trend",
            "date_column": date_col,
            "value_columns": list(value_cols),
            "insights": self._generate_trend_insights(df, date_col, value_cols)
        }
    
    def _generate_distribution_insights(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """生成分布洞察"""
        insights = []
        
        for col in columns:
            data = df[col].dropna()
            skewness = data.skew()
            
            if abs(skewness) > 1:
                direction = "右偏" if skewness > 0 else "左偏"
                insights.append(f"{col} 分布呈{direction}斜，建议考虑数据转换")
            
            # 检查异常值
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
            
            if len(outliers) > len(data) * 0.05:
                insights.append(f"{col} 包含较多异常值 ({len(outliers)} 个)，建议检查数据质量")
        
        return insights or ["所有特征分布相对正常"]
    
    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame) -> List[str]:
        """生成相关性洞察"""
        insights = []
        
        # 找出高相关性对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            insights.append(f"发现 {len(high_corr_pairs)} 对高相关性特征，可能存在多重共线性")
            for pair in high_corr_pairs[:3]:  # 显示前3对
                insights.append(f"  • {pair['var1']} 与 {pair['var2']}: {pair['correlation']:.3f}")
        else:
            insights.append("特征之间相关性适中，无明显多重共线性问题")
        
        return insights
    
    def _generate_comparison_insights(self, grouped_stats: pd.DataFrame, 
                                    group_by: str, value_col: str) -> List[str]:
        """生成对比洞察"""
        insights = []
        
        # 找出最高和最低组
        max_group = grouped_stats.loc[grouped_stats['mean'].idxmax()]
        min_group = grouped_stats.loc[grouped_stats['mean'].idxmin()]
        
        insights.append(f"{max_group[group_by]} 组的 {value_col} 平均值最高 ({max_group['mean']:.2f})")
        insights.append(f"{min_group[group_by]} 组的 {value_col} 平均值最低 ({min_group['mean']:.2f})")
        
        # 计算变异系数
        mean_val = grouped_stats['mean'].mean()
        std_val = grouped_stats['mean'].std()
        cv = std_val / mean_val if mean_val != 0 else 0
        
        if cv > 0.3:
            insights.append("不同组之间差异较大，分组变量可能是重要的预测因子")
        else:
            insights.append("不同组之间差异较小，分组变量的影响有限")
        
        return insights
    
    def _generate_trend_insights(self, df: pd.DataFrame, date_col: str, 
                               value_cols: List[str]) -> List[str]:
        """生成趋势洞察"""
        insights = []
        
        for col in value_cols:
            # 计算趋势
            data_sorted = df.sort_values(date_col)
            values = data_sorted[col].dropna()
            
            if len(values) > 10:
                # 简单线性趋势
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                if slope > 0:
                    insights.append(f"{col} 整体呈上升趋势")
                elif slope < 0:
                    insights.append(f"{col} 整体呈下降趋势")
                else:
                    insights.append(f"{col} 趋势相对平稳")
        
        return insights or ["趋势分析需要更多数据点"]
    
    def export_experiment_results(self, format: str = "json") -> str:
        """导出实验结果"""
        try:
            if format == "json":
                export_data = {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'total_experiments': len(self.experiments),
                    'experiments': [exp.to_dict() for exp in self.experiments]
                }
                
                export_path = self.output_dir / f"ml_experiments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                return str(export_path)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting experiment results: {str(e)}", exc_info=True)
            raise

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 1.5, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.exponential(1, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # 添加一些缺失值
    df.loc[np.random.choice(df.index, 50, replace=False), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'feature3'] = np.nan
    
    # 初始化工具
    ml_tools = EnhancedMLTools()
    
    print("=== 自动数据分析 ===")
    analysis_result = ml_tools.auto_data_analysis(df, target_column='target')
    print(f"数据质量分数: {analysis_result['data_quality_score']:.3f}")
    print(f"缺失值总数: {analysis_result['missing_values']['total_missing']}")
    
    print("\n=== 智能预处理 ===")
    processed_df, preprocessing_report = ml_tools.smart_preprocessing(df, target_column='target')
    print(f"处理后形状: {processed_df.shape}")
    print(f"应用的步骤数: {len(preprocessing_report['steps_applied'])}")
    
    print("\n=== 自动模型选择 ===")
    X = processed_df.drop('target', axis=1)
    y = processed_df['target']
    
    model_results = ml_tools.auto_model_selection(X, y)
    if 'best_model' in model_results and model_results['best_model']:
        best_model = model_results['best_model']
        print(f"最佳模型: {best_model['algorithm']}")
        print(f"准确率: {best_model['metrics']['accuracy']:.3f}")
    
    print("\n=== 创建可视化 ===")
    viz_result = ml_tools.create_advanced_visualizations(
        df, VisualizationType.DISTRIBUTION
    )
    if 'plot_path' in viz_result:
        print(f"分布图已保存到: {viz_result['plot_path']}")
    
    # 导出结果
    export_path = ml_tools.export_experiment_results()
    print(f"\n实验结果已导出到: {export_path}")