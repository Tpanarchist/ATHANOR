"""Native meaning conventions and teaching; neural interpretation is optional."""

from .lexicon import Binding, Lexicon, normalize
from .teaching import Example, Teacher

__all__ = ["Binding", "Example", "Lexicon", "Teacher", "normalize"]
