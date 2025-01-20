import torch
path = "./experiments/OCANetS1-celeba/models/last.ckpt"
save_path = "./celeba-S1.pth"

s=torch.load(path)

# for k,v in s.items():
#    print(k)

# for k,v in s["state_dict"].items():
#     if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
#         print(k)
# print(s["state_dict"])

new={}
for k,v in s["state_dict"].items():
    if "evaluator" not in k and "loss" not in k and "discriminator" not in k:
        k=k[10:]
        print(k)
        new[k]=v
    # print(k)

# for k,v in new.items():
#     print(k)
torch.save(new,save_path)
