from src.policies import *

"""
full square run 1 iter (useless): /home/andy/aa228/pz/results/winrate/2024_12_05_00:51:18

fast policies 100 iters (15 runs): /home/andy/aa228/pz/results/winrate/2024_12_05_01:25:59
fast policies 1000 iters (15 runs): /home/andy/aa228/pz/results/winrate/2024_12_05_01:47:30
"""

dirs = [
    "/home/andy/aa228/pz/results/winrate/2024_12_05_01:25:59", # fast policies 100 iters (15 files)
    "/home/andy/aa228/pz/results/winrate/2024_12_05_01:47:30", # fast policies 1000 iters (15 files)
    "/home/andy/aa228/pz/results/winrate/databricks" # slow policies run on databricks (7 files)
]

num_pairs = 0
for i, you_policy in enumerate(POLICIES):
    for opp_policy in POLICIES[i:]:
        num_files = 0
        for dir in dirs:
            filename = f"{you_policy}_vs_{opp_policy}.json"
            filename2 = f"{opp_policy}_vs_{you_policy}.json"
            files = os.listdir(dir)
            if filename in files or filename2 in files:
                # print(f"Found match {filename} in {dir}")
                num_files += 1
        print(you_policy, opp_policy, num_files)
        num_pairs += 1
print(num_pairs)