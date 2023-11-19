#successful
import tenseal as ts
import torch
import random

def generate_random_float_list(n):
    random_floats = [random.uniform(-5, 5) for _ in range(n)]
    return random_floats

def decrypt(enc):
    return enc.decrypt().tolist()

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192, #4096, 8192, 16384
            coeff_mod_bit_sizes=[60, 40, 40, 60] # [60, 40, 60], [60, 40, 40, 60], [60, 40, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

n = 2048
v1 = generate_random_float_list(n)
v2 = generate_random_float_list(n)

# encrypted vectors
enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

enc_result = enc_v1 + enc_v2 #addition
enc_avg = enc_result * 0.5

avg_decrypted = enc_avg.decrypt()
print(f'len(v1)={len(v1)}')
print(f'len(v2)={len(v2)}')
print(f'len(avg_decrypted)={len(avg_decrypted)}')

print(f'v1={v1[0:2]}/{v1[-2:]}')
print(f'v2={v2[0:2]}/{v2[-2:]}')
print(f'avg_decrypted={avg_decrypted[0:2]}/{avg_decrypted[-2:]}')

print(f'not equal:')
for i in range(len(avg_decrypted)):
    avg = (v1[i] + v2[i])/2.0
    if abs(avg - avg_decrypted[i]) > 0.01:
        print(f'[{i}],v1[{i}]={v1[i]},v2[{i}]={v2[i]},avg_decrypted[{i}]={avg_decrypted[i]},')

print(f'=========================================================================')
