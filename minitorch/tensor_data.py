from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    # TODO: Implement for Task 2.1.
    position = np.sum(np.multiply(index, strides))
    return position
    


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    # Assume strides s_i assends from left to right (in terms of thj array)
    ordinal_remainder = ordinal
    for i in range(len(shape))[::-1]:
        stride = np.prod(shape[:i])
        ordinal_quotient, ordinal_remainder = np.divmod(ordinal_remainder, stride)
        out_index[i] = ordinal_quotient
        
        


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    # TODO: Implement for Task 2.2.
    diff = len(big_shape) - len(shape)
    for i, d in enumerate(shape):
        out_index[i] = 0 if d == 1 else big_index[i + diff]
    return

def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    # TODO: Implement for Task 2.2.
    # Over verbose and not very robust implementation lmao
    # We do not deal with "funny rotations" i.e. adding dims of len 1 in the middle of the tensor
    # How to check for "implicit appending dimensions of 1 to the right"
    shorter_shape, longer_shape = (shape1, shape2) if len(shape1) <= len(shape2) else (shape2, shape1)
    output_shape = []
    diff = len(longer_shape) - len(shorter_shape)
    
    for i in range(len(longer_shape)):
        if i < diff:
            output_shape.append(longer_shape[i])
            continue
        # Catch mismatch
        if (shorter_shape[i-diff] == longer_shape[i]) or (shorter_shape[i-diff] == 1) or (longer_shape[i] == 1):
            output_shape.append(max(longer_shape[i], shorter_shape[i-diff]))
            continue
        raise IndexingError(f"Cannot broadcast, shapes {shape1} and {shape2} are incompatible")
    return tuple(output_shape)
        
                                                                    
def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Store the TensorData instance in CUDA memory"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a : first shape
            shape_b : second shape

        Returns:
        -------
            broadcasted shape

        Raises:
        ------
            IndexingError : if cannot broadcast

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Map a high-dimensional tensor index to the corresponding index in the underlying 1D array.

        This method converts a user-provided index for a multi-dimensional tensor into the 
        corresponding index of the underlying 1D array representation. It supports both 
        single integer indices for 1D tensors and tuple indices for higher-dimensional tensors.

        Args:
            index (Union[int, UserIndex]): The input index for the high-dimensional tensor.
                Can be an integer for 1D tensors or a tuple of integers for higher dimensions.

        Returns:
            int: The corresponding index in the underlying 1D array.

        Raises:
            IndexingError: If there's a mismatch between the shape of the input index and the tensor's shape.
            IndexingError: If any part of the index is out of range for the tensor's dimensions.
            IndexingError: If negative indexing is attempted (currently not supported).

        Notes:
            - For 0-dimensional tensors, the method treats them as 1-dimensional with a single element.
            - The method uses `index_to_position` for fast conversion of multi-dimensional indices.

        Example:
            >>> tensor = Tensor([1, 2, 3, 4], shape=(2, 2))
            >>> tensor.index((1, 1))
            3
            
        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns Tuple representation of the TensorData object

        Returns:
            Tuple[Storage, Shape, Strides]: Tensor Representation of storage, shape and strides
            
        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        # Approach: Permute order of strides
        shape: UserShape = tuple([self._shape[i] for i in order])
        strides: UserStrides = tuple([self._strides[i] for i in order])
        new_Tensor_Data = TensorData(self._storage, shape, strides)
        return new_Tensor_Data

    def to_string(self) -> str:
        """Generates String representation of current TensorData instance"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
