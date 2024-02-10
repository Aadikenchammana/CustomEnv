import json
import matplotlib.pyplot as plt
with open ("train_analytics//02.09.2024_09:12:16//rewardLog.txt","r") as f:
    log = f.read().splitlines()[1:]

def avg(lst):
    total = 0
    for item in lst:
        total += item
    return total/len(lst)
def ma(lst, interval):
    output = []
    ma_lst = []
    for item in lst:
        ma_lst.append(item)
        if len(ma_lst) > interval:
            ma_lst.pop(0)
        output.append(avg(ma_lst))
    return output

rewards = {}
current_episode = 0
for item in log:
    if item == " NEW TRAINING ITERATION CREATION":
        current_episode +=1
    else:
        r = float(item.split(",")[1][1:])
        if str(current_episode) not in rewards.keys():
            rewards[str(current_episode)] = [r]
        else:
            rewards[str(current_episode)] = rewards[str(current_episode)] + [r]
y = []
for episode in rewards.keys():
    lst = rewards[episode]
    r = lst[len(lst)-1]
    print(r)
    y.append(r)

plt.plot(ma(y,400))
plt.show()