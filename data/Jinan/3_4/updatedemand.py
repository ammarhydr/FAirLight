import json
import random

f = open('anon_3_4_jinan_real.json')

data = json.load(f)
# samples=random.sample(data, len(data)-3000)

blocks=['road_4_1_2', 'road_3_1_2', 'road_3_2_3', 'road_2_2_2', 'road_2_1_0']
# blocks=['road_4_1_2', 'road_3_1_2', 'road_3_2_3']
vehs=[]
for veh in data:
    if len(set(blocks) - set(veh['route'])) == len(blocks)  :
        vehs.append(veh)
    
# new_samples=data-samples
# new_samples=samples.copy()

with open("anon_3_4_jinan_real_new.json", "w") as outfile:
    json.dump(vehs, outfile)
    