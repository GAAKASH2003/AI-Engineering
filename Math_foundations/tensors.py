import numpy as np
from functools import reduce
from itertools import product as iterproduct

class Tensor:
    def __init__(self,data,shape=None):
        if isinstance(data,(list,tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data,np.ndarray):
            self._data=data.flatten().tolist()
            self._shape=tuple(data.shape)
        else:
            self._data=[data]
            self._shape=()

        if shape is not None:
            total=reduce(lambda a,b:a*b,shape,1)
            if total!=len(self._data):
                raise ValueError(
                      f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape=tuple(shape)
        self._strides= self._compute_strides(self._shape)
    
    def _flatten_nested(self,data):
        if not isinstance(data,(list,tuple)):
            return [data],()
        if len(data)==0:
            return [],(0,)
        
        sub_results=[self._flatten_nested(item) for item in data]
        sub_shape=sub_results[0][1]
        for i,(_, s) in enumerate(sub_results):
            if s!=sub_shape:
                raise ValueError(
                    f"Inconsistent shapes at index {i}: {s} vs {sub_shape}"
                )
                
        flat = []
        for sub_data, _ in sub_results:
            flat.extend(sub_data)
        
        return flat, (len(data),) + sub_shape
        
    @staticmethod
    def _compute_strides(shape):
        if(len(shape)==0):
            return ()
        strides=[1]*len(shape)
        for i in range(len(shape)-2,-1,-1):
            strides[i]=strides[i+1]*shape[i+1]
        
        return tuple(strides)

    @property
    def shape(self):
        return self._shape
    
    @property
    def rank(self):
        return len(self._shape)
    
    @property
    def size(self):
        return len(self._data)

    @property
    def strides(self):
        return self._strides
    
    def _flat_index(self,indices):
        if len(indices)!=len(self.shape):
            raise IndexError(
                f"Expected {len(self._shape)} indices, got {len(indices)}"
            )
        idx=0
        for i,(ind,stride) in enumerate(zip(indices,self._strides)):
            if ind < 0 or ind >= self._shape[i]:
                raise IndexError(
                    f"Index {ind} out of range for axis {i} with size {self._shape[i]}"
                )
            idx += ind * stride
        return idx
    
    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) == len(self._shape):
            return self._data[self._flat_index(indices)]
        raise IndexError("Partial indexing not supported in this basic implementation")    
    
    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self._data[self._flat_index(indices)] = value
        
    def reshape(self,new_shape):
        new_shape=list(new_shape)
        neg_idx=-1
        known_product=1
        for i,s in enumerate(new_shape):
            new_shape=list(new_shape)
            neg_idx=-1
            known_product=1
            
            for i,s in enumerate(new_shape):
                if s==-1:
                    if neg_idx != -1:
                        raise ValueError("Only one dimension can be -1")
                    neg_idx=i
                else:
                    known_product*=s
            
            if neg_idx != -1:
                new_shape[neg_idx] = self.size // known_product

            total = reduce(lambda a, b: a * b, new_shape, 1)
            if total != self.size:
                raise ValueError(
                    f"Cannot reshape {self.size} elements into shape {tuple(new_shape)}"
                )
                
        result = Tensor.__new__(Tensor)
        result._data = self._data[:]
        result._shape = tuple(new_shape)
        result._strides = self._compute_strides(result._shape)
        return result
                
    def squeeze(self, dim=None):
        if dim is not None:
            if self._shape[dim] != 1:
                return self.reshape(self._shape)
            new_shape = list(self._shape)
            new_shape.pop(dim)
            return self.reshape(tuple(new_shape) if new_shape else ())
        new_shape = tuple(s for s in self._shape if s != 1)
        if not new_shape:
            new_shape = ()
        return self.reshape(new_shape)
    
    def unsqueeze(self, dim):
        if dim < 0:
            dim = len(self._shape) + 1 + dim
        new_shape = list(self._shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))
    
    def transpose(self,dim0,dim1):
        perm=list(range(self.rank))
        perm[dim0],perm[dim1]=perm[dim1], perm[dim0]
        return self.permute(perm)
    
    def permute(self,dims):
        if sorted(dims)!=list(range(self.rank)):
            raise ValueError(f"Invalid permutation {dims} for rank {self.rank}")
        
        new_shape = tuple(self._shape[d] for d in dims)
        result = Tensor.__new__(Tensor)
        result._shape = new_shape
        result._strides = self._compute_strides(new_shape)
        result._data = [0] * self.size
        old_strides=self._strides
        for old_indices in iterproduct(*(range(s) for s in self._shape)):
            new_indices = tuple(old_indices[d] for d in dims)
            old_flat = sum(i * s for i, s in zip(old_indices, old_strides))
            new_flat = sum(
                i * s for i, s in zip(new_indices, result._strides)
            )
            result._data[new_flat] = self._data[old_flat]

        return result
    
    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim < 0:
            end_dim = self.rank + end_dim
        new_shape = (
            list(self._shape[:start_dim])
            + [reduce(lambda a, b: a * b, self._shape[start_dim:end_dim + 1], 1)]
            + list(self._shape[end_dim + 1:])
        )
        return self.reshape(tuple(new_shape))
    
    def _elementwise_op(self, other, op):
        if isinstance(other, (int, float)):
            result_data = [op(x, other) for x in self._data]
            return Tensor(result_data, shape=self._shape)
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type {type(other)}")
        if self._shape != other._shape:
            raise ValueError(
                f"Shape mismatch: {self._shape} vs {other._shape}. "
                "Use broadcast() first."
            )
        result_data = [op(a, b) for a, b in zip(self._data, other._data)]
        return Tensor(result_data, shape=self._shape)
    
    
    
    def __add__(self, other):
        return self._elementwise_op(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._elementwise_op(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self._elementwise_op(other, lambda a, b: a - b)
    
    def sum(self, axis=None):
        if axis is None:
            return sum(self._data)
        if axis < 0:
            axis = self.rank + axis
        new_shape = list(self._shape)
        axis_size = new_shape.pop(axis)

        result_size = reduce(lambda a, b: a * b, new_shape, 1)
        result_data = [0.0] * result_size
        result_strides = self._compute_strides(tuple(new_shape))

        for indices in iterproduct(*(range(s) for s in self._shape)):
            old_flat = sum(i * s for i, s in zip(indices, self._strides))
            new_indices = indices[:axis] + indices[axis + 1:]
            if new_indices:
                new_flat = sum(
                    i * s for i, s in zip(new_indices, result_strides)
                )
            else:
                new_flat = 0
            result_data[new_flat] += self._data[old_flat]

        if not new_shape:
            return result_data[0]
        return Tensor(result_data, shape=tuple(new_shape))

    def _build_nested(self, data, shape, offset):
        if len(shape) == 1:
            return data[offset:offset + shape[0]]
        result = []
        stride = reduce(lambda a, b: a * b, shape[1:], 1)
        for i in range(shape[0]):
            result.append(self._build_nested(data, shape[1:], offset + i * stride))
        return result

    def to_list(self):
        if not self._shape:
            return self._data[0]
        return self._build_nested(self._data, self._shape, 0)

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self.to_list()})"

    def to_numpy(self):
        return np.array(self._data).reshape(self._shape)
    



def basic_tensor():
    print("=" * 60)
    print("BASIC TENSOR OPERATIONS")
    print("=" * 60)
    
    scalar = Tensor(3.14)
    print(f"Scalar: shape={scalar.shape}, rank={scalar.rank}, value={scalar.to_list()}")

    vector = Tensor([1, 2, 3, 4, 5])
    print(f"Vector: shape={vector.shape}, rank={vector.rank}")
    
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Matrix: shape={matrix.shape}, rank={matrix.rank}")
    print(f"  matrix[0, 1] = {matrix[0, 1]}")
    print(f"  matrix[1, 2] = {matrix[1, 2]}")

    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"3D Tensor: shape={tensor_3d.shape}, rank={tensor_3d.rank}")
    print(f"  tensor[1, 0, 1] = {tensor_3d[1, 0, 1]}")

def demo_reshape_operations():
    print("="*60)
    print("RESHAPE OPERATIONS")
    print("="*60)
    
    data=Tensor(list(range(12)),shape=(2,6))  
    print(f"Original: shape={data.shape}")
    print(f"  {data.to_list()}")
    r1=data.reshape((3,4))
    print(f"\nReshaped to (3, 4): {r1.to_list()}")
    
    t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
    print(f"\nBefore squeeze: shape={t.shape}")
    s = t.squeeze()
    print(f"After squeeze():  shape={s.shape}")
    s0 = t.squeeze(dim=0)
    print(f"After squeeze(0): shape={s0.shape}")
# basic_tensor()
demo_reshape_operations()