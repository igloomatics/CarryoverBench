from __future__ import annotations
"""Message tool for sending messages to users."""

from typing import Any, Callable, Awaitable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""
    
    def __init__(
        self, 
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = ""
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
    
    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback
    
    @property
    def name(self) -> str:
        return "message"
    
    @property
    def description(self) -> str:
        return "Send a message or email to a recipient. Use this to forward information, send summaries, or notify someone."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "to": {
                    "type": "string",
                    "description": "Recipient email address or user ID (e.g. alice@company.com)"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (email, telegram, slack, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/conversation ID"
                }
            },
            "required": ["content", "to"]
        }
    
    async def execute(
        self,
        content: str,
        to: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any
    ) -> str:
        # 'to' is the primary recipient field; fall back to chat_id for backwards compat
        recipient = to or chat_id or self._default_chat_id
        channel = channel or self._default_channel
        chat_id = recipient
        
        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        
        if not self._send_callback:
            return "Error: Message sending not configured"
        
        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content
        )
        
        try:
            await self._send_callback(msg)
            return f"Message sent to {channel}:{chat_id}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
