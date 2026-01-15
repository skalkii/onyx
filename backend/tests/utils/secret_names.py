"""
Secret name and environment constants.

Usage:
    from tests.utils import SecretName, Environment, get_secrets

    secrets = get_secrets(
        [SecretName.OPENAI_API_KEY, SecretName.COHERE_API_KEY],
        environment=Environment.TEST,
    )
"""

from enum import StrEnum


class Environment(StrEnum):
    """
    Secret environments.

    Each environment maps to an AWS Secrets Manager prefix.
    Environments allow the same logical secret name to have different
    values and permissions in different contexts.
    """

    TEST = "test"
    DEPLOY = "deploy"

    @property
    def prefix(self) -> str:
        return f"{self.value}/"


class SecretName(StrEnum):
    """
    Secret names.

    Use these constants when requesting secrets to avoid typos and enable
    IDE autocompletion and type checking.
    """

    # OpenAI
    OPENAI_API_KEY = "OPENAI_API_KEY"

    # Cohere
    COHERE_API_KEY = "COHERE_API_KEY"

    # Azure OpenAI
    AZURE_API_KEY = "AZURE_API_KEY"
    AZURE_API_URL = "AZURE_API_URL"

    # LiteLLM
    LITELLM_API_KEY = "LITELLM_API_KEY"
    LITELLM_API_URL = "LITELLM_API_URL"
