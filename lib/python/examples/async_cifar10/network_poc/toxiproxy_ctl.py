# "remote control" for toxiproxy-server 
# every method sends an HTTP request that tells the Toxiproxy server to create a proxy, 
# add a network-degrading toxic to it, or delete it.
""""remote control" for toxiproxy-server. Every method sends an HTTP request that tells the Toxiproxy server to 
create a proxy, add a network-degrading toxic to it, or delete it"""

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
    # shared helper function
        url = f"{self.base_url}{path}"

        if body is not None:
            data = json.dumps(body).encode("utf-8")
        else:
            data = None

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
        return self._request(
            "POST",
            "/proxies",
            {"name": name, "listen": listen, "upstream": upstream}
        )

    def delete_proxy(self, name):
        return self._request(
            "DELETE",
            f"/proxies/{name}"
        )

    def add_toxic(self, proxy, name, type_, stream, attributes, toxicity=1.0):
    # POST /proxies/{proxy}/toxics
    return self._request("POST", f"/proxies/{proxy}/toxics",
                          {"name": name, "type": type_, "stream": stream,
                           "toxicity": toxicity, "attributes": attributes})

    def apply_profile(self, proxy, profile: NetworkProfile, stream: str):
        # stream is "upstream" (client->server) or "downstream" (server->client)
        # NOTE: Toxiproxy's Go backend rejects floats for these numeric
        # attributes, even whole numbers like 500.0 -- wrap in int().
        self.add_toxic(proxy, f"latency_{stream}", "latency", stream,
                        {"latency": int(profile.latency_ms), "jitter": int(profile.jitter_ms)})
        # bandwidth toxic "rate" is KB/s, not Mbps:
        # 1 Mbps = 1e6 bit/s = 125,000 B/s = 125 KB/s
        self.add_toxic(proxy, f"bandwidth_{stream}", "bandwidth", stream,
                        {"rate": int(profile.bandwidth_mbps * 125)})