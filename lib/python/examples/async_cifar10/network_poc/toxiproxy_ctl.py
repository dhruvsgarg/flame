""""remote control" for toxiproxy-server. Every method sends an HTTP request that tells the Toxiproxy server to 
create a proxy, add a network-degrading toxic to it, or delete it"""

"""Thin wrapper over the Toxiproxy HTTP API (default 127.0.0.1:8474).
Uses only the stdlib (urllib) -- no new dependency required.
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class NetworkProfile:
    latency_ms: float
    jitter_ms: float
    bandwidth_mbps: float


class ToxiproxyClient:
    def __init__(self, api_host="127.0.0.1", api_port=8474):
        self.base_url = f"http://{api_host}:{api_port}"

    def _request(self, method, path, body=None):
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} failed: {e.code} {e.read().decode()}")

    def create_proxy(self, name, listen, upstream):
        return self._request("POST", "/proxies",
                              {"name": name, "listen": listen, "upstream": upstream})

    def delete_proxy(self, name):
        return self._request("DELETE", f"/proxies/{name}")

    def add_toxic(self, proxy, name, type_, stream, attributes, toxicity=1.0):
        return self._request("POST", f"/proxies/{proxy}/toxics",
                              {"name": name, "type": type_, "stream": stream,
                               "toxicity": toxicity, "attributes": attributes})

    def apply_profile(self, proxy, profile: NetworkProfile, stream: str):
        self.add_toxic(proxy, f"latency_{stream}", "latency", stream,
                        {"latency": int(profile.latency_ms), "jitter": int(profile.jitter_ms)})
        self.add_toxic(proxy, f"bandwidth_{stream}", "bandwidth", stream,
                        {"rate": int(profile.bandwidth_mbps * 125)})