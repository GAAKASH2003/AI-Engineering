class Value:
    def __init__(self,data,children=(),op=''):
        self.data=data
        self.grad=0.0
        self._backward=lambda:None
        self._prev=set(children)
        self._op=op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += (other.data*out.grad)
            other.grad += (self.data*out.grad)
        out._backward=_backward 
        return out
    
    def relu(self):
        out=Value(max(0,self.data),(self,),'relu')
        def _backward():
            self.grad+= (1 if out.data>0 else 0.0)*out.grad
        out._backward=_backward
        return out
                
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for v in reversed(topo):
            v._backward()   


def demo_basic():
    print("=== Basic: y = relu(x1 * x2 + 1) ===")
    x1=Value(2.0)  
    x2=Value(3.0)
    a=x1*x2
    b=a+Value(1.0)
    y=b.relu()
    y.backward()
    
    print(f"  x1 = 2.0, x2 = 3.0")
    print(f"  y  = {y.data}")
    print(f"  dy/dx1 = {x1.grad}  (expected 3.0 = x2)")
    print(f"  dy/dx2 = {x2.grad}  (expected 2.0 = x1)")


demo_basic()