from dataclasses import dataclass
from typing import ClassVar, Self
from .defaults import q
import numpy as np

@dataclass(frozen=True)
class ModInt:
    """
    A class that represents integers modulo an integer p i.e. x mod p.
    """
    value: int
    p: int

    def __post_init__(self):
        object.__setattr__(self, "value", self.value % self.p)

    def __is_ring_element(self, other: object) -> object | None:
        # Checks if the other object is a ModInt and
        # if it belongs to the same ring (i.e., has the same modulus)
        if isinstance(other, int):
            return ModInt(other, self.p)
        if not isinstance(other, ModInt):
            raise TypeError("Elements must be ModInts.")
        if self.p != other.p:
            raise ValueError("Elements must belong to the same ring.")
        return other
    
    def __add__(self, other: Self) -> Self:
        """
        Add two elements mod p.
        """
        other = self.__is_ring_element(other)
        return type(self)(self.value + other.value, self.p)

    def __radd__(self, other: Self) -> Self:
        """
        Right addition to support int + ModInt.
        """
        return self.__add__(other)
    
    def __sub__(self, other: Self) -> Self:
        """
        Subtract two elements mod p.
        """
        other = self.__is_ring_element(other)
        return type(self)(self.value - other.value, self.p)
    
    def __mul__(self, other: Self) -> Self:
        """
        Multiply two elements mod p.
        """
        other = self.__is_ring_element(other)
        return type(self)(self.value * other.value, self.p)
    
    def __rmul__(self, other: Self) -> Self:
        """
        Right multiplication to support int * ModInt.
        """
        return self.__mul__(other)
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two modulo p elements are equal.
        """
        if not isinstance(other, type(self)):
            return False
        return self.value == other.value
    
    def __neg__(self) -> Self:
        """
        Negate a modulo p element.
        """
        return type(self)(-self.value, self.p)
    
    def __lshift__(self, other: object) -> Self:
        """
        Left shift a modulo p element by a non-negative integer.
        """
        if not isinstance(other, int) or other < 0:
            raise ValueError("Shift amount must be a non-negative integer.")
        return type(self)(self.value << other, self.p)
    
    def __rshift__(self, other: object) -> Self:
        """
        Right shift a modulo p element by a non-negative integer.
        """
        if not isinstance(other, int) or other < 0:
            raise ValueError("Shift amount must be a non-negative integer.")
        return type(self)(self.value >> other, self.p)
    
    def __and__(self, other: Self) -> Self:
        """
        Bitwise AND of two modulo p elements.
        """
        other = self.__is_ring_element(other)
        return type(self)(self.value & other.value, self.p)

# These classes inherits from ModInt which supports basic ring operations.
@dataclass(frozen=True)    
class RingElement(ModInt):
    """
    A class that represents an element of the ring of integers modulo an integer p.
    """
    p: ClassVar[int] = q

    def __init__(self, value: int, p: int = p):
        if p is None:
            p = self.p
        super().__init__(value, p)
    
    @staticmethod
    def get_ring_vector(vec: np.ndarray[int], p: int = p) -> np.ndarray[object]:
        """
        Get the vector containing ring element
        """
        # Convert the input vector to a numpy array of RingElements
        return np.array([RingElement(int(x), p) for x in vec], dtype=object)
    
    @staticmethod
    def extract_normal_vector(vec: np.ndarray[object]) -> np.ndarray[int]:
        """
        Extract the normal vector from a vector of RingElements.
        """
        # print("Extracting normal vector from RingElement vector:", vec[0])
        return np.array([x.value for x in vec], dtype=int)

    def __repr__(self) -> str:
        return f"{self.value}"
