import numpy as np
import json

with open("pathfinder-path.json", "r") as f:
    p=json.load(f)
print(len(p))
p1=np.array(p[: :12])
print(p1.shape)
print(p1)

g=np.random.random(p1.shape)


r=p1+g*0.05
print(r.shape)
print(r)

with open("test_tracking2.json", "wt") as f:
    s=json.dump(p1.tolist(),f)
with open("test_tracking3.json", "wt") as f:
    s=json.dump(r.tolist(),f)


# np.save("test_tracking2.npy",arr=p1)
# np.save("test_tracking3.npy",arr=r)
