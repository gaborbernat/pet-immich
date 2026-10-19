from typing import TYPE_CHECKING

import httpx
import pytest

from pet_immich.network import local_request

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:2283",
        "http://localhost:2283",
        "https://10.2.0.143",
        "http://172.16.0.1",
        "http://192.168.1.1",
        "http://[::1]",
        "http://[fd00::1]",
    ],
)
def test_local_destinations_reach_transport(url: str, mocker: MockerFixture) -> None:
    handler = mocker.create_autospec(httpx.BaseTransport().handle_request, return_value=httpx.Response(200))
    with httpx.Client(transport=httpx.MockTransport(handler), event_hooks={"request": [local_request]}) as client:
        response = client.get(url)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://8.8.8.8",
        "http://169.254.169.254",
        "http://0.0.0.0",
        "http://192.0.2.1",
        "http://[2001:4860:4860::8888]",
        "ftp://127.0.0.1",
    ],
)
def test_external_destinations_never_reach_transport(url: str, mocker: MockerFixture) -> None:
    handler = mocker.create_autospec(httpx.BaseTransport().handle_request, return_value=httpx.Response(200))
    with (
        httpx.Client(transport=httpx.MockTransport(handler), event_hooks={"request": [local_request]}) as client,
        pytest.raises(ValueError, match="Photo"),
    ):
        client.post(url, content=b"private-crop")
    handler.assert_not_called()


def test_redirect_cannot_send_data_to_external_host(mocker: MockerFixture) -> None:
    handler = mocker.create_autospec(
        httpx.BaseTransport().handle_request, return_value=httpx.Response(307, headers={"location": "https://8.8.8.8"})
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [local_request]},
            follow_redirects=True,
        ) as client,
        pytest.raises(ValueError, match="external"),
    ):
        client.post("http://127.0.0.1", content=b"private-crop")
    assert handler.call_count == 1
