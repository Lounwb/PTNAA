import subprocess

commands = []
i = 0
with open('adv_classification_results.txt', 'r') as f:
    adv_labels = [line.split(":") for line in f.readlines()] 

with open('classification_results.txt', 'r') as f:
    labels = [line.split(":") for line in f.readlines()] 
p1 = 0
p2 = 0
while p1 < len(labels) and p2 < len(adv_labels):
    name = labels[p1][0].split('.')[0].strip()
    label = labels[p1][1].strip()

    adv_name = adv_labels[p2][0].split('.')[0].strip()
    adv_label = adv_labels[p2][1].strip()
    command = f"python dmbp_exp.py -m resnet50 -i /home/qqq/data/lty/FIA/dataset/images/{name}.png -tl {label} -sp /home/qqq/data/lty/FIA/adv/PTNAAPIDI/res-v2/{adv_name}.png -sl {adv_label}"
    commands.append(command)
    p1 += 1
    p2 += 1

for cmd in commands:
    i = i + 1
    if i%100 == 0:
        print('Executing commands f{i}')
    subprocess.run(cmd, shell=True)

print('Commands Executed over...')