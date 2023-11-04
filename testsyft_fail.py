#https://blog.openmined.org/ckks-homomorphic-encryption-pytorch-pysyft-seal/
import syft as sy

import torch as th

import syft.frameworks.tenseal as ts



# hook PyTorch to add extra functionalities like the ability to encrypt torch tensors

hook = sy.TorchHook(th)



# Generate CKKS public and secret keys

public_keys, secret_key = ts.generate_ckks_keys()





matrix = th.tensor([[10.5, 73, 65.2], [13.33, 22, 81]])



matrix_encrypted = matrix.encrypt("ckks", public_key=public_keys)



# to use for plain evaluations

t_eval = th.tensor([[1, 2.5, 4], [13, 7, 16]])

# to use for encrypted evaluations

t_encrypted = t_eval.encrypt("ckks", public_key=public_keys)



print("encrypted tensor + plain tensor")

result = matrix_encrypted + t_eval

# result is an encrypted tensor

print(result.decrypt(secret_key=secret_key))



print("encrypted tensor + encrypted tensor")

result = matrix_encrypted + t_encrypted

# result is an encrypted tensor

print(result.decrypt(secret_key=secret_key))



print("encrypted tensor - plain tensor")

result = matrix_encrypted - t_eval

# result is an encrypted tensor

print(result.decrypt(secret_key=secret_key))



print("encrypted tensor - encrypted tensor")

result = matrix_encrypted - t_encrypted

# result is an encrypted tensor

print(result.decrypt(secret_key=secret_key))



print("encrypted tensor * plain tensor")

result = matrix_encrypted * t_eval

# result is an encrypted tensor

print(result.decrypt(secret_key=secret_key))



print("encrypted tensor * encrypted tensor")

result = matrix_encrypted * t_encrypted

# result is an encrypted tensor

print(result.decrypt(secret_key=secret_key))