import random
import json;

def generate_addition_data(num_samples=1000,n=3,mode=0):
    dataset = []
    for _ in range(num_samples):
        # 随机决定位数，让模型适应不同长度
        digits_a = random.randint(1, n)
        digits_b = random.randint(1, n)

        a = random.randint(0, 10 ** digits_a - 1)
        b = random.randint(0, 10 ** digits_b - 1)
        result = a + b

        # 构造对话格式
        sample = {
            "input": f"{str(a).zfill(n)}+{str(b).zfill(n)}=",
            "output": f"{str(result).zfill(n+1)}"
        }
        # sample = {
        #     "input": f"{a}+{b}=",
        #     "output": f"{result}"
        # }
        dataset.append(sample)
    name = ["dataset","test","val"]

    with open("%s.json"%name[mode], "w") as f:
        json.dump(dataset, f);
n=5;
num_train=2500
generate_addition_data(num_train,n)
generate_addition_data(100,n,mode=1)
generate_addition_data(50,n,mode=2)
print("start")

