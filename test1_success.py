#https://www.cnblogs.com/pam-sh/p/16026065.html
import tenseal as ts
import numpy as np

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

plain1 = ts.plain_tensor([1,2,3,4], [2,2])
print(" First tensor: Shape = {} Data = {}".format(plain1.shape, plain1.tolist()))

plain2 = ts.plain_tensor(np.array([5,6,7,8]).reshape(2,2))
print(" Second tensor: Shape = {} Data = {}".format(plain2.shape, plain2.tolist()))

encrypted_tensor1 = ts.ckks_tensor(context, plain1)
encrypted_tensor2 = ts.ckks_tensor(context, plain2)

print(" Shape = {}".format(encrypted_tensor1.shape))
print(" Encrypted Data = {}.".format(encrypted_tensor1))


encrypted_tensor_from_np = ts.ckks_tensor(context, np.array([5,6,7,8]).reshape([2,2]))
print(" Shape = {}".format(encrypted_tensor_from_np.shape))

# 输出：
# First tensor: Shape = [2, 2] Data = [[1.0, 2.0], [3.0, 4.0]]
# Second tensor: Shape = [2, 2] Data = [[5.0, 6.0], [7.0, 8.0]]
# Shape = [2, 2]
# Encrypted Data = <tenseal.tensors.ckkstensor.CKKSTensor object at 0x7f9ddd530400>.
# Shape = [2, 2]