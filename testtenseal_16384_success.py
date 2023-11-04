#successful
import tenseal as ts
import torch
# MLP(
#   (layer_input): Linear(in_features=60, out_features=200, bias=True) => 12000+200
#   (relu): ReLU()
#   (layer_hidden): Linear(in_features=200, out_features=10, bias=True) => 2000+10
# )
# total params: 14210 < 16384
# # Setup TenSEAL context with poly_modulus_degree=16384
# context = ts.context(
#             ts.SCHEME_TYPE.CKKS,
#             poly_modulus_degree=16384,
#             coeff_mod_bit_sizes=[60, 40, 40, 60, 60]
#           )
# context.generate_galois_keys()
# context.global_scale = 2**40

def decrypt(enc):
    return enc.decrypt().tolist()

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

v1 = [0, 1, 2, 3, 4]
v2 = [4, 3, 2, 1, 0]

# encrypted vectors
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

result = enc_v1 + enc_v2 #addition
avg = result * 0.5
print(result.decrypt()) # ~ [4, 4, 4, 4, 4], correct
print(avg.decrypt()) # ~ [2, 2, 2, 2, 2], correct

result = enc_v1.dot(enc_v2) # dot product
print(result.decrypt()) # ~ [10], wrong

matrix = [
  [73, 0.5, 8],
  [81, -5, 66],
  [-100, -78, -2],
  [0, 9, 17],
  [69, 11 , 10],
]
result = enc_v1.matmul(matrix) #matrix multiplication
print(result.decrypt()) # ~ [157, -90, 153], wrong
###################################################################
print('ckks tensors1:')
plain1 = ts.plain_tensor([[0, 1], [3, 4]])
plain2 = ts.plain_tensor([[4, 3], [1, 0]])

# encrypted tensors
encrypted_tensor1 = ts.ckks_tensor(context, plain1)
encrypted_tensor2 = ts.ckks_tensor(context, plain2)

result = encrypted_tensor1 + encrypted_tensor2 #addition
print(decrypt(result)) # ~ [4, 4, 4, 4, 4], correct
#============================
print('ckks tensors2:')
plain1 = ts.plain_tensor(torch.tensor([[0, 1], [3, 4]]))
plain2 = ts.plain_tensor(torch.tensor([[4, 3], [1, 0]]))

# encrypted tensors
encrypted_tensor1 = ts.ckks_tensor(context, plain1)
encrypted_tensor2 = ts.ckks_tensor(context, plain2)

result = encrypted_tensor1 + encrypted_tensor2 #addition
print(decrypt(result)) # ~ [4, 4, 4, 4, 4], correct
#============================
print('ckks tensors3 (MLP case):')
plain1 = torch.tensor([[0, 1], [3, 4]])
plain2 = torch.tensor([[4, 3], [1, 0]])

# encrypted tensors
encrypted_tensor1 = ts.ckks_tensor(context, plain1)
encrypted_tensor2 = ts.ckks_tensor(context, plain2)

result = encrypted_tensor1 + encrypted_tensor2 #addition
print(decrypt(result)) # ~ [4, 4, 4, 4, 4], correct
#============================
print('ckks tensors4:')
plain1 = [[0, 1], [3, 4]]
plain2 = [[4, 3], [1, 0]]

# encrypted tensors
encrypted_tensor1 = ts.ckks_tensor(context, plain1)
encrypted_tensor2 = ts.ckks_tensor(context, plain2)

result = encrypted_tensor1 + encrypted_tensor2 #addition
print(decrypt(result)) # ~ [4, 4, 4, 4, 4], correct