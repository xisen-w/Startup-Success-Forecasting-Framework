import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 如果没有请 pip install seaborn

# 1. 定义状态和示例历史数据序列
states = ["Commute", "Work", "Meeting", "Lunch", "Break"]
historical = [
    "Commute", "Work", "Work", "Meeting", "Work", "Lunch", "Work",
    "Break", "Work", "Commute", "Work", "Meeting", "Work",
    "Lunch", "Work", "Break", "Commute"
]

# 2. 计算转移计数和转移概率矩阵
state_index = {s: i for i, s in enumerate(states)}
n = len(states)
counts = np.zeros((n, n), dtype=int)

for a, b in zip(historical, historical[1:]):
    counts[state_index[a], state_index[b]] += 1

prob = np.zeros_like(counts, dtype=float)
for i in range(n):
    total = counts[i].sum()
    if total > 0:
        prob[i, :] = counts[i, :] / total

transition_df = pd.DataFrame(prob, index=states, columns=states)

# 3. 在终端打印，并保存到 CSV
print("状态转移概率矩阵：")
print(transition_df.to_markdown())
transition_df.to_csv("transition_matrix.csv", encoding="utf-8-sig")

# 4. 可视化：热力图
plt.figure(figsize=(8, 6))
sns.heatmap(transition_df, annot=True, fmt=".2f", cmap="Blues")
plt.title("一阶马尔可夫链 转移概率矩阵 Heatmap")
plt.ylabel("当前状态")
plt.xlabel("下一状态")
plt.tight_layout()
plt.savefig("transition_matrix_heatmap.png")
plt.show()

# 5. 模拟新的事件序列并可视化
start_state = "Commute"
sim_length = 12
sim = [start_state]
for _ in range(sim_length - 1):
    current = sim[-1]
    next_state = np.random.choice(states, p=prob[state_index[current]])
    sim.append(next_state)

plt.figure(figsize=(8, 4))
y = [state_index[s] for s in sim]
plt.plot(y, marker='o')
plt.yticks(range(len(states)), states)
plt.xlabel("时间步")
plt.ylabel("事件状态")
plt.title("基于一阶马尔可夫链的日历事件模拟")
plt.tight_layout()
plt.savefig("simulated_sequence.png")
plt.show()