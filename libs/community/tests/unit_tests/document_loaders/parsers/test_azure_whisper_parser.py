"""Tests for the Azure OpenAI Whisper parser."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser


@pytest.mark.requires("openai")
@patch("openai.AzureOpenAI")
def test_azure_openai_whisper(mock_client: MagicMock) -> None:
    endpoint = "endpoint"
    key = "key"
    version = "115"
    name = "model"

    parser = AzureOpenAIWhisperParser(
        api_key=key, azure_endpoint=endpoint, api_version=version, deployment_name=name
    )
    mock_client.assert_called_once_with(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=version,
        max_retries=3,
        azure_ad_token=None,
    )
    assert parser._client == mock_client()
