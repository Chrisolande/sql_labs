from tenacity import retry, stop_after_attempt, wait_exponential_jitter


def default_retry(initial: float = 2, max_wait: float = 60, attempts: int = 5):
    return retry(
        wait=wait_exponential_jitter(initial=initial, max=max_wait),
        stop=stop_after_attempt(attempts),
    )
