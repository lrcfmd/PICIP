from typing import Collection, TypeAlias

from pymatgen.core import Composition

KnownsLike: TypeAlias = Composition | Collection[Composition]