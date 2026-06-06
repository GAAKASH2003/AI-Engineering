import math
import random

def check_convexity(f,dim,bounds=(-5,5),samples=2000,label=""):
    violations=0
    worst_violation=0
    
    for _ in range(samples):
        x=[random.uniform(*bounds) for _ in range(dim)]
        y=[random.uniform(*bounds) for _ in range(dim)]
        t=random.uniform(0,1)
        mid=[t*xi+(1-t)*yi for xi,yi in zip(x,y)]
        lhs=f(mid)
        rhs=t*f(x)+(1-t)*f(y)
        gap=lhs-rhs
        if gap>1e-10:
            violations+=1
            worst_violation=max(worst_violation,gap)
   
    is_convex=violations==0
    status="CONVEX" if is_convex else "NOT_CONVEX"
    if label:
       print(f"  {label:30s}  {status:10s}  violations: {violations}/{samples}"
             + (f"  worst: {worst_violation:.6f}" if violations > 0 else ""))
    
    return is_convex,violations


         
def hessian_eigenvalues_2d(H):
    a,b=H[0][0],H[0][1]
    c,d=H[1][0],H[1][1]
    
    trace=a+d
    det=a*d-b*c
    discriminant=(trace**2-4*det)
    if discriminant<0:
        return None,None
    sqrt_disc=math.sqrt(discriminant)
    e1=(trace+sqrt_disc)/2
    e2=(trace-sqrt_disc)/2
    return e1,e2

    
def is_positive_semidefinite_2d(H):
    e1, e2 = hessian_eigenvalues_2d(H)
    if e1 is None or e2 is None:
        return False
    return e1 >= -1e-10 and e2 >= -1e-10 

def invert_2X2(H):
    det=(H[0][0]*H[1][1]-H[0][1]*H[1][0])
    if abs(det)<1e-15:
        return None

    return [
        [H[1][1]/det,-H[0][1]/det],
        [-H[1][0]/det,-H[0][0]/det]
    ]

def mat_vec_2d(M,v):
    return [
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1],
    ]

class GradientDescent:
    def __init__(self,lr=0.001):
        self.lr=lr
    
    def step(self,params,grads):
        return [p-self.lr*g for p,g in zip(params,grads)]
    

def optimize_gd(grad_f,x0,lr=0.01,steps=1000,tol=1e-12):
    x=list(x0)
    history=[x[:]]
    for _ in range(steps):
        g=grad_f(x)
        if sum(gi**2 for gi in g)<tol:
            break
        x=[xi-lr*gi for xi,gi in zip(x,g)]
        if any(math.isnan(xi) or math.isinf(xi) for xi in x):
            break
        history.append(x[:])
    return history

def newtons_method(grad_f,hessian_f,x0,steps=100,tol=1e-12):
    x=list(x0)
    history=[x[:]]
    for _ in range(steps):
        g=grad_f(x)
        if sum(gi**2 for gi in g)<tol:
            break
        H=hessian_f(x)
        H_inv=invert_2X2(H)
        if H_inv is None:
            break
        
        dx=mat_vec_2d(H_inv,g)
        x=[x[0]-dx[0],x[1]-dx[1]]
        if any(math.isnan(xi) or math.isinf(xi) for xi in x):
            break
        history.append(x[:])
        
    return history

        
        
def demo_convexity_checker():
    print("=" * 65)
    print("  CONVEXITY CHECKER")
    print("=" * 65)
    print()

    random.seed(42)

    check_convexity(lambda x: x[0] ** 2, 1, label="f(x) = x^2")
    check_convexity(lambda x: abs(x[0]), 1, label="f(x) = |x|")
    check_convexity(lambda x: math.exp(x[0]), 1, label="f(x) = e^x")
    check_convexity(lambda x: x[0] ** 2 + x[1] ** 2, 2, label="f(x,y) = x^2 + y^2")
    check_convexity(lambda x: max(x[0], 0), 1, label="f(x) = max(0, x) [ReLU]")

    check_convexity(lambda x: math.sin(x[0]), 1, label="f(x) = sin(x)")
    check_convexity(lambda x: x[0] ** 3, 1, label="f(x) = x^3")
    check_convexity(lambda x: -x[0] ** 2, 1, label="f(x) = -x^2")
    check_convexity(
        lambda x: math.sin(x[0]) * math.cos(x[1]),
        2,
        label="f(x,y) = sin(x)*cos(y)"
    )
    check_convexity(
        lambda x: x[0] ** 2 - x[1] ** 2,
        2,
        label="f(x,y) = x^2 - y^2 [saddle]"
    )

    print()
    print("  Top group: expected convex. Bottom group: expected non-convex.")



def demo_hessian_analysis():
    print()
    print()
    print("=" * 65)
    print("  HESSIAN ANALYSIS AND CURVATURE")
    print("=" * 65)

    print()
    print("  f(x,y) = 5x^2 + y^2 (elongated bowl)")
    H1 = [[10, 0], [0, 2]]
    e1, e2 = hessian_eigenvalues_2d(H1)
    psd = is_positive_semidefinite_2d(H1)
    print(f"  Hessian: [[{H1[0][0]}, {H1[0][1]}], [{H1[1][0]}, {H1[1][1]}]]")
    print(f"  Eigenvalues: {e1:.1f}, {e2:.1f}")
    print(f"  Positive semidefinite: {psd}")
    print(f"  Convex: {psd}")

    print()
    print("  f(x,y) = x^2 - y^2 (saddle)")
    H2 = [[2, 0], [0, -2]]
    e1, e2 = hessian_eigenvalues_2d(H2)
    psd = is_positive_semidefinite_2d(H2)
    print(f"  Hessian: [[{H2[0][0]}, {H2[0][1]}], [{H2[1][0]}, {H2[1][1]}]]")
    print(f"  Eigenvalues: {e1:.1f}, {e2:.1f}")
    print(f"  Positive semidefinite: {psd}")
    print(f"  Saddle point: mixed signs confirm saddle")

    print()
    print("  f(x,y) = x^2 + 3xy + y^2")
    H3 = [[2, 3], [3, 2]]
    e1, e2 = hessian_eigenvalues_2d(H3)
    psd = is_positive_semidefinite_2d(H3)
    print(f"  Hessian: [[{H3[0][0]}, {H3[0][1]}], [{H3[1][0]}, {H3[1][1]}]]")
    print(f"  Eigenvalues: {e1:.1f}, {e2:.1f}")
    print(f"  Positive semidefinite: {psd}")
    print(f"  Convex: {psd} (negative eigenvalue means indefinite)")

    print()
    print("  Rosenbrock at minimum (1, 1)")
    Hmin = [[802, -400], [-400, 200]]
    e1, e2 = hessian_eigenvalues_2d(Hmin)
    psd = is_positive_semidefinite_2d(Hmin)
    print(f"  Hessian: [[{Hmin[0][0]}, {Hmin[0][1]}], [{Hmin[1][0]}, {Hmin[1][1]}]]")
    print(f"  Eigenvalues: {e1:.2f}, {e2:.2f}")
    print(f"  Positive semidefinite at (1,1): {psd}")


# def demo_newtons_method():
#     print()
#     print()
#     print("=" * 65)
#     print("  NEWTON'S METHOD vs GRADIENT DESCENT")
#     print("=" * 65)

#     def f(x):
#         return 50 * x[0] ** 2 + x[1] ** 2

#     def grad_f(x):
#         return [100 * x[0], 2 * x[1]]

#     def hessian_f(x):
#         return [[100, 0], [0, 2]]

#     start = [10.0, 10.0]

#     print()
#     print(f"  Function: f(x,y) = 50x^2 + y^2")
#     print(f"  Minimum at: (0, 0), f = 0")
#     print(f"  Starting point: ({start[0]}, {start[1]}), f = {f(start):.1f}")
#     print(f"  Condition number: {100 / 2:.0f} (elongated valley)")

#     newton_hist = newtons_method(grad_f, hessian_f, start, steps=50)
#     gd_hist = optimize_gd(grad_f, start, lr=0.015, steps=500)

#     print()
#     print(f"  Newton's method: {len(newton_hist) - 1} steps to converge")
#     print(f"  {'Step':>6s}  {'x':>12s}  {'y':>12s}  {'f(x,y)':>14s}")
#     print(f"  {'-' * 48}")
#     for i, p in enumerate(newton_hist):
#         print(f"  {i:6d}  {p[0]:12.8f}  {p[1]:12.8f}  {f(p):14.8f}")

#     print()
#     threshold = 1e-10
#     gd_converged = len(gd_hist) - 1
#     for i, p in enumerate(gd_hist):
#         if f(p) < threshold:
#             gd_converged = i
#             break

#     print(f"  Gradient descent (lr=0.015): {len(gd_hist) - 1} steps taken")
#     steps_to_show = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 499]
#     steps_to_show = [s for s in steps_to_show if s < len(gd_hist)]
#     print(f"  {'Step':>6s}  {'x':>12s}  {'y':>12s}  {'f(x,y)':>14s}")
#     print(f"  {'-' * 48}")
#     for i in steps_to_show:
#         p = gd_hist[i]
#         print(f"  {i:6d}  {p[0]:12.8f}  {p[1]:12.8f}  {f(p):14.8f}")

#     print()
#     print(f"  Newton converged in {len(newton_hist) - 1} step(s)")
#     print(f"  GD reached f < {threshold} at step {gd_converged}"
#           + (" (did not converge)" if gd_converged == len(gd_hist) - 1 else ""))
#     print()
#     print("  Newton's method is exact for quadratic functions.")
#     print("  GD struggles with high condition number (elongated valleys).")



demo_convexity_checker()
demo_hessian_analysis()

