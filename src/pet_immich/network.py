"""Keep photo requests on explicit local addresses and out of environment proxies."""

from ipaddress import ip_address, ip_network
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


def local_request(request: httpx.Request) -> None:
    """Reject public destinations before sending credentials, crops, or asset identifiers."""
    host = request.url.host
    if host == "localhost":
        request.url = request.url.copy_with(host="127.0.0.1")
        host = "127.0.0.1"
    try:
        address = ip_address(host)
    except ValueError as error:
        msg = "Photo services require a literal local IP address or localhost"
        raise ValueError(msg) from error
    if request.url.scheme not in {"http", "https"} or not (
        address.is_loopback
        or any(
            address in ip_network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
        )
    ):
        msg = "Photo requests to external services are disabled"
        raise ValueError(msg)


__all__ = ["local_request"]
