"""Explicit public failures: rejected work never publishes partial knowledge."""


class AthanorError(Exception):
    """Base class for a rejected structural action."""


class ValidationError(AthanorError):
    """A value, definition, reference, or result violates its contract."""


class ExecutionError(AthanorError):
    """A well-formed program cannot execute on these inputs."""


class ResourceLimitError(ExecutionError):
    """The invocation exceeded its explicit work or call-depth budget."""


class PersistenceError(AthanorError):
    """A snapshot cannot be encoded, validated, saved, or restored."""
