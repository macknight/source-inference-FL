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

def decrypt(enc):
    return enc.decrypt().tolist()

plain1 = ts.plain_tensor([1,2,3,4], [2,2])
print("First tensor: Shape = {} Data = {}".format(plain1.shape, plain1.tolist()))

plain2 = ts.plain_tensor(np.array([5,6,7,8]).reshape(2,2))
print("Second tensor: Shape = {} Data = {}".format(plain2.shape, plain2.tolist()))

encrypted_tensor1 = ts.ckks_tensor(context, plain1)
encrypted_tensor2 = ts.ckks_tensor(context, plain2)

#密文（张量）+ 密文（张量）
result = encrypted_tensor1 + encrypted_tensor2
print("Plain equivalent: {} + {}\nDecrypted result: {}.".format(plain1.tolist(), plain2.tolist(), decrypt(result)))

#密文（张量）- 密文（张量）
result = encrypted_tensor1 - encrypted_tensor2
print("Plain equivalent: {} - {}\nDecrypted result: {}.".format(plain1.tolist(), plain2.tolist(), decrypt(result)))

#密文（张量）* 密文（张量）
result = encrypted_tensor1 * encrypted_tensor2
print("Plain equivalent: {} * {}\nDecrypted result: {}.".format(plain1.tolist(), plain2.tolist(), decrypt(result)))

#密文（张量）* 明文（张量）
plain = ts.plain_tensor([5,6,7,8], [2,2])
result = encrypted_tensor1 * plain
print("Plain equivalent: {} * {}\nDecrypted result: {}.".format(plain1.tolist(), plain.tolist(), decrypt(result)))

#取反：密文（张量）
result = -encrypted_tensor1 
print("Plain equivalent: -{}\nDecrypted result: {}.".format(plain1.tolist(), decrypt(result)))

#求幂：密文（张量）^3
result = encrypted_tensor1 ** 3
print("Plain equivalent: {} ^ 3\nDecrypted result: {}.".format(plain1.tolist(), decrypt(result)))

#多项式计算（整数）：1 + X^2 + X^3，X是密文（张量）
result = encrypted_tensor1.polyval([1,0,1,1])
print("X = {}".format(plain1.tolist()))
print("1 + X^2 + X^3 = {}.".format(decrypt(result)))

#多项式计算（浮点数）0.5 + 0.197X^2 - 0.004X^3，X是密文（张量）
result = encrypted_tensor1.polyval([0.5, 0.197, 0, -0.004])
print("X = {}".format(plain1.tolist()))
print("0.5 + 0.197 X - 0.004 X^3 = {}.".format(decrypt(result)))

# 输出：
# First tensor: Shape = [2, 2] Data = [[1.0, 2.0], [3.0, 4.0]]
# Second tensor: Shape = [2, 2] Data = [[5.0, 6.0], [7.0, 8.0]]
# Plain equivalent: [[1.0, 2.0], [3.0, 4.0]] + [[5.0, 6.0], [7.0, 8.0]]
# Decrypted result: [[6.000000000510762, 7.99999999944109], [10.000000000176103, 11.999999999918177]].
# Plain equivalent: [[1.0, 2.0], [3.0, 4.0]] - [[5.0, 6.0], [7.0, 8.0]]
# Decrypted result: [[-3.999999998000314, -3.9999999987240265], [-4.0000000013643, -4.0000000013791075]].
# Plain equivalent: [[1.0, 2.0], [3.0, 4.0]] * [[5.0, 6.0], [7.0, 8.0]]
# Decrypted result: [[5.000000678675058, 12.000001612431278], [21.000002812898412, 32.000004287986336]].
# Plain equivalent: [[1.0, 2.0], [3.0, 4.0]] * [[5.0, 6.0], [7.0, 8.0]]
# Decrypted result: [[5.000000676956037, 12.000001612473657], [21.000002810086173, 32.00000428474004]].
# Plain equivalent: -[[1.0, 2.0], [3.0, 4.0]]
# Decrypted result: [[-1.0000000012552241, -2.000000000358531], [-2.9999999994059015, -3.999999999269536]].
# Plain equivalent: [[1.0, 2.0], [3.0, 4.0]] ^ 3
# Decrypted result: [[1.0000008094463497, 8.000006439159353], [27.000021714154222, 64.00005146475934]].
# X = [[1.0, 2.0], [3.0, 4.0]]
# 1 + X^2 + X^3 = [[3.000000945752252, 13.000006978595758], [37.00002291844665, 81.000053606697]].
# X = [[1.0, 2.0], [3.0, 4.0]]
# 0.5 + 0.197 X - 0.004 x^X = [[0.6930000194866153, 0.8620000226394146], [0.9829999914891329, 1.0319998662943677]].