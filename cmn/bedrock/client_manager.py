"""
cmn/bedrock/client_manager.py
=============================
Factory for creating Bedrock boto3 clients with optimal configurations.

Usage
-----
    from cmn.bedrock.client_manager import BedrockClientFactory
    
    # Create a bedrock-runtime client
    bedrock_client = BedrockClientFactory.bedrock_runtime(region='us-east-1')
    
    # Use in a cached Streamlit function
    import streamlit as st
    
    @st.cache_resource
    def get_bedrock_client(region):
        return BedrockClientFactory.bedrock_runtime(region)
"""

import boto3
from botocore.config import Config


class BedrockClientFactory:
    """Factory for creating Bedrock boto3 clients with production-ready configuration."""

    # Default configuration for all Bedrock clients
    _DEFAULT_CONFIG = Config(
        max_pool_connections=10,
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        connect_timeout=5,
        read_timeout=60,
    )

    @classmethod
    def bedrock_runtime(cls, region: str, config: Config = None) -> object:
        """
        Create a Bedrock runtime client with connection pooling and retry logic.

        Parameters
        ----------
        region : str
            AWS region (e.g., 'us-east-1', 'us-west-2')
        config : botocore.config.Config, optional
            Custom Config object. Defaults to production-ready settings.

        Returns
        -------
        boto3.client
            A bedrock-runtime client with:
            - Connection pooling (max 10 connections)
            - Adaptive retry strategy (3 attempts)
            - 5s connect timeout, 60s read timeout
        """
        if config is None:
            config = cls._DEFAULT_CONFIG

        return boto3.client(
            'bedrock-runtime',
            region_name=region,
            config=config,
        )

    @classmethod
    def bedrock_agents_runtime(cls, region: str, config: Config = None) -> object:
        """
        Create a Bedrock Agents runtime client.

        Parameters
        ----------
        region : str
            AWS region
        config : botocore.config.Config, optional
            Custom Config object.

        Returns
        -------
        boto3.client
            A bedrock-agents-runtime client
        """
        if config is None:
            config = cls._DEFAULT_CONFIG

        return boto3.client(
            'bedrock-agent-runtime',
            region_name=region,
            config=config,
        )

    @classmethod
    def bedrock_models(cls, region: str, config: Config = None) -> object:
        """
        Create a Bedrock Models client (for listing models, etc.).

        Parameters
        ----------
        region : str
            AWS region
        config : botocore.config.Config, optional
            Custom Config object.

        Returns
        -------
        boto3.client
            A bedrock client (for model management)
        """
        if config is None:
            config = cls._DEFAULT_CONFIG

        return boto3.client(
            'bedrock',
            region_name=region,
            config=config,
        )

    @classmethod
    def with_custom_config(
        cls,
        service: str,
        region: str,
        max_pool_connections: int = 10,
        connect_timeout: int = 5,
        read_timeout: int = 60,
        max_retries: int = 3,
    ) -> object:
        """
        Create a Bedrock client with custom connection settings.

        Parameters
        ----------
        service : str
            Service name ('bedrock-runtime', 'bedrock-agent-runtime', 'bedrock')
        region : str
            AWS region
        max_pool_connections : int
            Maximum number of connections in pool (default: 10)
        connect_timeout : int
            Connection timeout in seconds (default: 5)
        read_timeout : int
            Read timeout in seconds (default: 60)
        max_retries : int
            Maximum retry attempts (default: 3)

        Returns
        -------
        boto3.client
            A configured Bedrock client
        """
        config = Config(
            max_pool_connections=max_pool_connections,
            retries={'max_attempts': max_retries, 'mode': 'adaptive'},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        return boto3.client(
            service,
            region_name=region,
            config=config,
        )
