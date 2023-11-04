import syft as sy
import tenseal as ts
import torch

# 创建 PySyft hook 和本地工作机,from syft0.3.0
local_worker = sy.VirtualMachine(name="local_worker")

# 创建 TenSEAL 上下文和密钥对
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[60, 40, 40, 60]) #error: ValueError: encryption parameters are not set correctly
encoder = ts.encoder(context)
public_key, secret_key = ts.keygen(context)
encryptor = ts.encryptor(context, public_key)

# 创建一个 PyTorch 张量
x = torch.tensor([1, 2, 3, 4])

# 将 PyTorch 张量转换为 TenSEAL 密文
x_encrypted = ts.ckks_vector(encoder.encode(x))
y_encrypted = x_encrypted.square()  # 例如，计算平方，这是一个同态运算

# 解密结果
result = y_encrypted.decrypt(secret_key)
print(result)
