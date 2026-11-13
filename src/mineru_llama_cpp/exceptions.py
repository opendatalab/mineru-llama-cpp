"""Exception hierarchy for mineru_llama_cpp.

See design spec §5.3. The C++ binding layer (src/cpp/binding.cpp) imports
this module and raises these exact classes when llama-server reports an
error — do not rename these classes without updating binding.cpp's
raise_mapped_error() to match.
"""


class MineruLlamaCppError(Exception):
    """Base class for all exceptions raised by this library."""

    def __init__(
        self,
        *args: object,
        error_type: str | None = None,
        code: int | str | None = None,
        details: dict | None = None,
    ) -> None:
        """保留原始类型、错误码和完整服务响应，避免上层只能按消息猜测分类。"""
        super().__init__(*args)
        self.error_type = error_type
        self.code = code
        self.details = dict(details or {})


class InvalidRequestError(MineruLlamaCppError):
    """The request itself was invalid: malformed JSON, bad image data, etc.

    Corresponds to llama-server's error_type::ERROR_TYPE_INVALID_REQUEST
    (JSON field "type": "invalid_request_error"), or to a raw C++ exception
    raised before the request was ever posted to the engine (e.g. malformed
    body JSON).
    """


class ContextExceededError(InvalidRequestError):
    """The prompt (plus requested completion length) exceeds n_ctx.

    Corresponds to llama-server's error_type::ERROR_TYPE_EXCEED_CONTEXT_SIZE
    (JSON field "type": "exceed_context_size_error"). Subclasses
    InvalidRequestError because it is, semantically, a client-supplied input
    that was too large for the configured Engine — callers that only care
    about "was my request bad" can catch InvalidRequestError and this is
    included; callers that specifically want to detect and react to
    context-size overflow (e.g. by truncating and retrying) can catch this
    subclass directly.
    """


class EngineError(MineruLlamaCppError):
    """An internal engine error occurred (not the caller's fault).

    Corresponds to every other llama-server error_type (server_error,
    not_found_error, permission_error, authentication_error,
    not_supported_error, unavailable_error) or to an unrecognized/missing
    "type" field.
    """


class EngineUnavailableError(EngineError):
    """已确认后台循环退出，此实例永久不可复用；普通 EngineError 不代表死亡。"""


class EngineClosedError(EngineUnavailableError):
    """调用方已关闭实例；后续请求须使用显式新建的引擎。"""
