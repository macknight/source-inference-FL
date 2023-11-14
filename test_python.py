w_params_noised = [[1,2],[3,4],[5,6]]  # 这是您的动态列表

# 计算平均值
avg_w_params_noised = [sum(values) / len(values) for values in zip(*w_params_noised)]

print(avg_w_params_noised)