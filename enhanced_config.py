"""
Enhanced Configuration Manager
增强配置管理器 - 支持多API供应商和动态配置
"""

import yaml
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """API供应商类型"""
    GOOGLE = "google"
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"

@dataclass
class ProviderConfig:
    """API供应商配置"""
    name: str
    type: ProviderType
    enabled: bool
    models: List[str]
    default_model: str
    api_key: str
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['type'] = self.type.value
        return result

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.providers = self._load_providers()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                # 替换环境变量
                config = self._replace_env_vars(config)
                return config
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
                
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return self._get_default_config()
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """递归替换环境变量"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, "")
        else:
            return obj
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "default_provider": "google",
            "providers": {
                "google": {
                    "name": "Google Gemini",
                    "type": "google",
                    "enabled": True,
                    "models": ["gemini-2.5-flash-preview-05-20"],
                    "default_model": "gemini-2.5-flash-preview-05-20",
                    "api_key": os.getenv("GOOGLE_API_KEY", "")
                }
            },
            "features": {
                "enhanced_agent": True,
                "memory_system": True,
                "task_planning": True,
                "resource_monitoring": True,
                "ml_tools": True
            }
        }
    
    def _load_providers(self) -> Dict[str, ProviderConfig]:
        """加载供应商配置"""
        providers = {}
        
        for provider_id, provider_data in self.config.get("providers", {}).items():
            try:
                provider_config = ProviderConfig(
                    name=provider_data.get("name", provider_id),
                    type=ProviderType(provider_data.get("type", "openai")),
                    enabled=provider_data.get("enabled", True),
                    models=provider_data.get("models", []),
                    default_model=provider_data.get("default_model", ""),
                    api_key=provider_data.get("api_key", ""),
                    base_url=provider_data.get("base_url"),
                    api_version=provider_data.get("api_version"),
                    deployment_name=provider_data.get("deployment_name")
                )
                providers[provider_id] = provider_config
                
            except Exception as e:
                logger.error(f"Error loading provider {provider_id}: {str(e)}")
                continue
                
        return providers
    
    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """获取启用的供应商"""
        return {k: v for k, v in self.providers.items() if v.enabled and v.api_key}
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """获取指定供应商配置"""
        return self.providers.get(provider_id)
    
    def get_default_provider(self) -> Optional[ProviderConfig]:
        """获取默认供应商"""
        default_id = self.config.get("default_provider", "google")
        return self.get_provider(default_id)
    
    def update_provider(self, provider_id: str, updates: Dict[str, Any]):
        """更新供应商配置"""
        if provider_id in self.providers:
            provider = self.providers[provider_id]
            
            # 更新字段
            if "api_key" in updates:
                provider.api_key = updates["api_key"]
            if "base_url" in updates:
                provider.base_url = updates["base_url"]
            if "default_model" in updates:
                provider.default_model = updates["default_model"]
            if "enabled" in updates:
                provider.enabled = updates["enabled"]
                
            logger.info(f"Updated provider {provider_id}")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            # 更新配置字典
            self.config["providers"] = {
                k: v.to_dict() for k, v in self.providers.items()
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
            logger.info(f"Config saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def get_feature_setting(self, feature: str, default: bool = True) -> bool:
        """获取功能开关设置"""
        return self.config.get("features", {}).get(feature, default)
    
    def get_app_settings(self) -> Dict[str, Any]:
        """获取应用设置"""
        return self.config.get("app_settings", {
            "title": "🤖 Enhanced AutoML Workflow Agent",
            "description": "具备智能记忆、任务规划和增强推理能力的AI助手",
            "page_icon": "🤖",
            "layout": "wide",
            "sidebar_state": "expanded"
        })
    
    def get_resource_monitoring_settings(self) -> Dict[str, Any]:
        """获取资源监控设置"""
        return self.config.get("resource_monitoring", {
            "enabled": True,
            "interval_seconds": 5,
            "cpu_threshold": 80,
            "memory_threshold": 80,
            "disk_threshold": 85
        })

def render_provider_config_ui(config_manager: ConfigManager):
    """渲染供应商配置UI"""
    st.subheader("🔧 API供应商配置")
    
    # 获取所有供应商
    providers = config_manager.providers
    enabled_providers = config_manager.get_enabled_providers()
    
    # 当前选择的供应商
    if "selected_provider" not in st.session_state:
        default_provider = config_manager.get_default_provider()
        st.session_state.selected_provider = list(enabled_providers.keys())[0] if enabled_providers else "google"
    
    # 供应商选择
    provider_options = {k: f"{v.name} ({'✅' if v.enabled and v.api_key else '❌'})" 
                       for k, v in providers.items()}
    
    selected_provider_id = st.selectbox(
        "选择API供应商",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.selected_provider) 
        if st.session_state.selected_provider in provider_options else 0
    )
    
    st.session_state.selected_provider = selected_provider_id
    provider = providers[selected_provider_id]
    
    # 配置表单
    with st.form(f"provider_config_{selected_provider_id}"):
        st.write(f"**配置 {provider.name}**")
        
        # 启用/禁用
        enabled = st.checkbox("启用此供应商", value=provider.enabled)
        
        # API Key
        api_key = st.text_input(
            "API Key",
            value=provider.api_key,
            type="password",
            help="从供应商处获取的API密钥"
        )
        
        # 模型选择
        if provider.models:
            default_model = st.selectbox(
                "默认模型",
                options=provider.models,
                index=provider.models.index(provider.default_model) 
                if provider.default_model in provider.models else 0
            )
        else:
            default_model = st.text_input("默认模型", value=provider.default_model)
        
        # 特定配置
        base_url = None
        if provider.type in [ProviderType.OPENAI, ProviderType.AZURE, ProviderType.OPENAI_COMPATIBLE]:
            base_url = st.text_input(
                "Base URL",
                value=provider.base_url or "",
                help="API基础URL（如：https://api.openai.com/v1）"
            )
        
        # Azure特定配置
        api_version = None
        deployment_name = None
        if provider.type == ProviderType.AZURE:
            api_version = st.text_input(
                "API版本",
                value=provider.api_version or "2023-12-01-preview"
            )
            deployment_name = st.text_input(
                "部署名称",
                value=provider.deployment_name or ""
            )
        
        # 保存按钮
        if st.form_submit_button("保存配置"):
            updates = {
                "enabled": enabled,
                "api_key": api_key,
                "default_model": default_model
            }
            
            if base_url is not None:
                updates["base_url"] = base_url
            if api_version is not None:
                updates["api_version"] = api_version
            if deployment_name is not None:
                updates["deployment_name"] = deployment_name
            
            config_manager.update_provider(selected_provider_id, updates)
            st.success(f"✅ {provider.name} 配置已保存")
            st.rerun()
    
    # 测试连接
    if provider.enabled and provider.api_key:
        if st.button(f"🔗 测试 {provider.name} 连接"):
            success = test_provider_connection(provider)
            if success:
                st.success("✅ 连接测试成功！")
            else:
                st.error("❌ 连接测试失败，请检查配置")
    
    # 显示当前状态
    st.subheader("📊 供应商状态")
    for provider_id, provider_config in providers.items():
        status = "🟢" if provider_config.enabled and provider_config.api_key else "🔴"
        st.write(f"{status} **{provider_config.name}**: "
                f"{'已启用' if provider_config.enabled else '已禁用'}, "
                f"{'已配置' if provider_config.api_key else '未配置'} API Key")

def test_provider_connection(provider: ProviderConfig) -> bool:
    """测试供应商连接"""
    try:
        if provider.type == ProviderType.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=provider.default_model,
                google_api_key=provider.api_key
            )
            # 简单测试
            response = llm.invoke("测试连接")
            return True
            
        elif provider.type == ProviderType.OPENAI:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=provider.default_model,
                api_key=provider.api_key,
                base_url=provider.base_url
            )
            response = llm.invoke("测试连接")
            return True
            
        # 其他供应商的测试逻辑...
        
    except Exception as e:
        logger.error(f"Provider connection test failed: {str(e)}")
        return False
    
    return False

def create_llm_from_config(provider: ProviderConfig):
    """根据配置创建LLM实例"""
    try:
        if provider.type == ProviderType.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=provider.default_model,
                google_api_key=provider.api_key
            )
            
        elif provider.type == ProviderType.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=provider.default_model,
                api_key=provider.api_key,
                base_url=provider.base_url
            )
            
        elif provider.type == ProviderType.AZURE:
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                deployment_name=provider.deployment_name,
                api_key=provider.api_key,
                api_version=provider.api_version,
                azure_endpoint=provider.base_url
            )
            
        elif provider.type == ProviderType.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=provider.default_model,
                api_key=provider.api_key
            )
            
        elif provider.type == ProviderType.OPENAI_COMPATIBLE:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=provider.default_model,
                api_key=provider.api_key,
                base_url=provider.base_url
            )
            
    except Exception as e:
        logger.error(f"Error creating LLM from config: {str(e)}")
        raise

# 全局配置管理器实例
config_manager = ConfigManager()