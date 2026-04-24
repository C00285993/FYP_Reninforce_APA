"""
Shared API error handling — provides clean, user-friendly error messages
for Anthropic API failures instead of exposing raw exception details.

Usage:
    from utils.api_error_handler import handle_api_error

    try:
        response = client.messages.create(...)
    except Exception as e:
        handle_api_error(e, logger, context="payload generation")
"""

import logging


def classify_api_error(exc: Exception) -> tuple[str, str]:
    """
    Classify an API exception into a user-friendly message and fix suggestion.

    Returns:
        (message, fix) — both are short, human-readable strings.
    """
    err = str(exc).lower()

    if "credit" in err or "billing" in err or "balance" in err:
        return (
            "Anthropic API credit balance is empty.",
            "Top up at https://console.anthropic.com/settings/billing "
            "or use --no-llm-crawl / --no-llm-payloads to skip LLM features.",
        )

    if "api_key" in err or "authentication" in err or "401" in err or "could not resolve auth" in err:
        return (
            "Invalid or missing Anthropic API key.",
            "Check your ANTHROPIC_API_KEY environment variable "
            "or use --no-llm-crawl / --no-llm-payloads.",
        )

    if "rate" in err and "limit" in err or "429" in err:
        return (
            "Anthropic API rate limit reached.",
            "Wait a moment and try again, or reduce concurrency.",
        )

    if "connection" in err or "timeout" in err or "unreachable" in err:
        return (
            "Cannot reach the Anthropic API.",
            "Check your internet connection and try again.",
        )

    return (
        "Anthropic API request failed.",
        "Check your API key and network connection. "
        "Use --no-llm-crawl / --no-llm-payloads to skip LLM features.",
    )


_shown_messages: set[str] = set()  # module-global: suppress repeats across instances


def handle_api_error(
    exc: Exception,
    logger: logging.Logger,
    *,
    context: str = "LLM call",
    once_flag_obj: object | None = None,
    once_flag_attr: str = "_api_error_shown",
) -> None:
    """
    Log a clean, one-time warning for an API error.

    The message is suppressed if the same (context, error class) pair has
    already been shown in this process — prevents spam when multiple
    env instances hit the same billing/auth error.

    Args:
        exc:            The caught exception.
        logger:         Logger to write to.
        context:        Short label for what was being attempted (e.g. "payload generation").
        once_flag_obj:  Object to set a flag on so the message only prints once.
                        Pass `self` for class methods. If None, always logs.
        once_flag_attr: Attribute name for the once-flag.
    """
    message, fix = classify_api_error(exc)

    # Global dedup: only show each (context-category, error-type) once per process
    dedup_key = f"{message[:30]}"
    if dedup_key in _shown_messages:
        return
    _shown_messages.add(dedup_key)

    # Per-instance dedup (legacy)
    if once_flag_obj is not None:
        if getattr(once_flag_obj, once_flag_attr, False):
            return
        setattr(once_flag_obj, once_flag_attr, True)

    logger.warning("%s disabled -- %s %s", context.capitalize(), message, fix)
