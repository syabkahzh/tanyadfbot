from __future__ import annotations

import asyncio

from tools.hermes_control import send_command


class _FakeDatabase:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_hermes_command(self, command: str, payload: str) -> int:
        self.sent.append((command, payload))
        return 123


def test_hermes_control_rejects_unknown_command(capsys) -> None:
    db = _FakeDatabase()

    result = asyncio.run(send_command(db, "drop_database", "{}"))

    captured = capsys.readouterr()
    assert result == 2
    assert "Unsupported command" in captured.err
    assert db.sent == []
