"""
Secrets utilities for fetching test secrets.

Secrets are resolved in order:
    1. Environment variables (already set in the process)
    2. .env file at the repo root (loaded via dotenv)
    3. AWS Secrets Manager (batch fetch for any remaining keys)

Usage:
    In conftest.py, set up a session-scoped fixture:

        @pytest.fixture(scope="session")
        def test_secrets() -> dict[SecretName, str]:
            return get_secrets(
                [SecretName.OPENAI_API_KEY, SecretName.COHERE_API_KEY],
                environment=Environment.TEST,
            )

    Then use in test fixtures:

        @pytest.fixture
        def openai_client(test_secrets: dict[SecretName, str]) -> OpenAI:
            return OpenAI(api_key=test_secrets[SecretName.OPENAI_API_KEY])

Configuration via OS environment variables:
    - AWS_REGION: AWS region for Secrets Manager (default: "us-east-1")

AWS SSO Authentication:
    boto3 automatically uses SSO credentials if configured in ~/.aws/config.
    Run `aws sso login` to authenticate before running tests.
"""

import logging
import os

from dotenv import dotenv_values

from tests.utils.secret_names import Environment
from tests.utils.secret_names import SecretName

logger = logging.getLogger(__name__)

# AWS Secrets Manager configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Path to the .env file used by tests (relative to repo root)
_DOTENV_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, ".vscode", ".env"
)


def _get_local_secrets(keys: list[SecretName]) -> dict[SecretName, str]:
    """
    Resolve secrets from environment variables and the .env file.

    Checks os.environ first, then falls back to values in .vscode/.env.
    Returns only the keys that were found locally.
    """
    dotenv = dotenv_values(_DOTENV_PATH)
    found: dict[SecretName, str] = {}

    for key in keys:
        # os.environ takes precedence
        value = os.environ.get(key.value) or dotenv.get(key.value)
        if value:
            found[key] = value

    return found


def _get_aws_secrets(
    keys: list[SecretName],
    environment: Environment,
) -> dict[SecretName, str]:
    """
    Fetch secrets from AWS Secrets Manager in a single batch request.
    """
    import boto3
    from botocore.exceptions import ClientError

    prefix = environment.prefix

    session = boto3.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=AWS_REGION,
    )

    secret_ids = [f"{prefix}{name}" for name in keys]

    try:
        response = client.batch_get_secret_value(SecretIdList=secret_ids)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "AccessDeniedException":
            raise RuntimeError(
                f"Access denied to secrets with prefix '{prefix}'. "
                f"Please check your AWS credentials/permissions or run 'aws sso login'."
            ) from e
        elif error_code == "UnrecognizedClientException":
            raise RuntimeError(
                "AWS credentials not found or expired. "
                "If using SSO, run 'aws sso login' to authenticate."
            ) from e
        else:
            raise RuntimeError(
                f"Failed to fetch secrets from AWS Secrets Manager: {e}"
            ) from e

    secrets: dict[SecretName, str] = {}
    for secret in response.get("SecretValues", []):
        secret_id = secret.get("Name", "")
        secret_value = secret.get("SecretString")

        if secret_value:
            key_name = (
                secret_id[len(prefix) :] if secret_id.startswith(prefix) else secret_id
            )
            try:
                secrets[SecretName(key_name)] = secret_value
            except ValueError:
                logger.warning(f"Secret '{key_name}' not in SecretName enum, skipping")

    for error in response.get("Errors", []):
        secret_id = error.get("SecretId", "unknown")
        error_code = error.get("ErrorCode", "unknown")
        message = error.get("Message", "unknown error")
        logger.warning(
            f"Failed to fetch secret '{secret_id}': [{error_code}] {message}"
        )

    return secrets


def get_secrets(
    keys: list[SecretName],
    environment: Environment = Environment.TEST,
) -> dict[SecretName, str]:
    """
    Resolve secrets from local sources first, then AWS Secrets Manager.

    Checks environment variables and .vscode/.env before making any AWS calls.
    Only keys not found locally are fetched from AWS.

    Args:
        keys: List of secret names to resolve.
        environment: The AWS environment to fetch from (default: Environment.TEST).

    Returns:
        dict: Mapping of SecretName to secret values.

    Raises:
        RuntimeError: If AWS secrets cannot be fetched due to auth/access issues.
    """
    if not keys:
        return {}

    secrets = _get_local_secrets(keys)

    if secrets:
        local_names = ", ".join(k.value for k in secrets)
        logger.info(f"Resolved {len(secrets)} secret(s) locally: {local_names}")

    remaining: list[SecretName] = [k for k in keys if k not in secrets]
    if remaining:
        aws_secrets = _get_aws_secrets(remaining, environment)
        secrets.update(aws_secrets)
        logger.info(
            f"Fetched {len(aws_secrets)}/{len(remaining)} secret(s) from AWS "
            f"(environment: {environment})"
        )

    return secrets
