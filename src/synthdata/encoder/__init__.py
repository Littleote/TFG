from ._functions import auto
from ._functions import greater
from ._functions import lower

from ._classes import EncoderNone as none
from ._classes import EncoderIgnore as ignore
from ._classes import EncoderLimit as limit
from ._classes import EncoderOHE as OHE
from ._classes import EncoderEquivalence as Equivalence

__all__ = ['auto', 'greater', 'lower', 'none', 'ignore', 'limit', 'OHE', 'Equivalence']