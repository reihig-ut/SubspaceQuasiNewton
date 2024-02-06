from algorithms.descent_method import BFGS
import jax.numpy as jnp
from utils.calculate import subspace_line_search,jax_randn,clipping_eigenvalues

class SubspaceQNM(BFGS):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.Pk = None
    self.Hk = None
    self.projected_gradk = None
    self.params_key = [
      "alpha",
      "beta",
      "reduced_dim",
      "matrix_size",
      "dim",
      "backward",
      "lower_eigenvalue",
      "upper_eigenvalue"
    ]
  
  def __run_init__(self, f, x0, iteration,params):
    super().__run_init__(f, x0, iteration,params)
    dim = params["dim"]
    matrix_size = params["matrix_size"]
    reduced_dim = params["reduced_dim"]
    
    self.Hk = jnp.eye(matrix_size,dtype = self.dtype)
    Q = self.generate_matrix(dim=dim,
                             reduced_dim=reduced_dim,
                             mode = "random")
    random_projected_grad = self.subspace_first_order_oracle(x = self.xk,Mk = Q)
    self.update_Pk(matrix_size,
                   random_projected_grad,
                   Q)
    self.projected_gradk = self.subspace_first_order_oracle(self.xk,self.Pk)

  def __direction__(self, projected_grad):
    return -self.Hk@projected_grad
  
  def __step_size__(self, projected_grad,dk,Mk,params):
    alpha = params["alpha"]
    beta = params["beta"]
    return subspace_line_search(self.xk,self.f,projected_grad=projected_grad,dk=dk,Mk=Mk,alpha=alpha,beta=beta)
  

  def __iter_per__(self, params):
    reduced_dim = params["reduced_dim"]
    dim = params["dim"]
    lower_eigenvalue = params["lower_eigenvalue"]
    upper_eigenvalue = params["upper_eigenvalue"]
    matrix_size = params["matrix_size"]
    
    dk = self.__direction__(self.projected_gradk)
    s = self.__step_size__(projected_grad=self.projected_gradk,
                           dk=dk,
                           Mk = self.Pk,
                           params=params)
    self.__update__(s*self.Pk.T@dk)
    projected_gradk1 = self.subspace_first_order_oracle(self.xk,self.Pk)
    self.update_BFGS(sk = s*dk,yk = projected_gradk1 - self.projected_gradk)
    self.Hk = clipping_eigenvalues(self.Hk,lower=lower_eigenvalue,upper=upper_eigenvalue)
    Q = self.generate_matrix(dim=dim,
                             reduced_dim=reduced_dim,
                             mode = "random")
    random_projected_grad = self.subspace_first_order_oracle(self.xk,Q)
    self.update_Pk(matrix_size,
                   random_projected_grad,
                   Q)
    self.projected_gradk = projected_gradk1

  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return jax_randn(reduced_dim,dim,dtype=self.dtype)
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")

  def update_Pk(self,matrix_size,random_projected_grad,Qk):
    dim = Qk.shape[1]
    # P^\top = [x_0/||x_0||,QTQ\nabla f(x_0)/||QTQ\nabla f(x_0)||,...,x_k/||x_k||,QTQ\nabla f(x_k)/||QTQ\nabla f(x_k)||]
    random_projected_grad_fullsize = Qk.T@random_projected_grad
    if jnp.linalg.norm(self.xk) < 1e-12:
      vector1 = jnp.zeros((1,dim),dtype=self.dtype)
    else:
      vector1 = jnp.expand_dims(self.xk,0)/jnp.linalg.norm(self.xk)
    
    if jnp.linalg.norm(random_projected_grad_fullsize) < 1e-12:
      vector2 = jnp.zeros((1,dim),dtype=self.dtype)
    else:
      vector2 = jnp.expand_dims(random_projected_grad_fullsize ,0)/jnp.linalg.norm(random_projected_grad_fullsize)
    if self.Pk is None:
      self.Pk = jnp.concatenate([jnp.eye(matrix_size-2,dim,dtype = self.dtype),vector1,vector2])
    else:
      self.Pk = jnp.concatenate([self.Pk[2:],vector1,vector2])

    