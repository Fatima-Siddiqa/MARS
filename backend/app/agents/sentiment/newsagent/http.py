import requests

from app.agents.sentiment.newsagent import config

_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT, "Accept-Language": "en"})


def get(url, **kwargs):
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT)
    response = _session.get(url, **kwargs)
    response.raise_for_status()
    return response


def post(url, **kwargs):
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT)
    response = _session.post(url, **kwargs)
    response.raise_for_status()
    return response
