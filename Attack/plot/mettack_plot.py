import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns

plt.figure(figsize=(17,15))

# 加粗边框
bwith = 5  # 边框宽度设置为2
ax = plt.gca()  # 获取边框
ax.spines['top'].set_visible(True)  # 去掉上边框
ax.spines['right'].set_visible(True)  # 去掉右边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
# sns.set(style="darkgrid")

# name = "nettack_cora"
# name = "nettack_citeseer"
name = "pubmed"
df = pd.read_excel("./attack.xlsx", sheet_name=name)
# data = df.values
# df.set_index('ptb_rate',inplace=True)
print(df)
x = df.iloc[:,0:1].to_numpy()
only_rough_denoising = df.iloc[:,1:2].to_numpy()
only_fine_denoising= df.iloc[:,2:3].to_numpy()
LC_GCN = df.iloc[:,-1:].to_numpy()
print(df.keys())
print(LC_GCN)
plt.plot(x,df.iloc[:, 1], marker='<',markersize=20,color="blue", label=df.keys()[1], linestyle="--",linewidth=5 )
plt.plot(x,df.iloc[:, 2], marker='o',markersize=20,color="orange", label=df.keys()[2], linestyle="--",linewidth=5 )
plt.plot(x,df.iloc[:, 3], marker='*',markersize=20,color="red", label=df.keys()[3], linestyle="--",linewidth=5 )
plt.plot(x,df.iloc[:, 4], marker='+',markersize=20,color="pink", label=df.keys()[4], linestyle="--",linewidth=5 )
plt.plot(x,df.iloc[:, 5], marker='x',markersize=20,color="purple", label=df.keys()[5], linestyle="--",linewidth=5 )
plt.plot(x,df.iloc[:, 6], marker='h',markersize=20,color="brown", label=df.keys()[6], linestyle="--",linewidth=5 )
# plt.plot(x,df.iloc[:, 7], marker='v',markersize=20,color="grey", label=df.keys()[7], linestyle="--",linewidth=5 )
# plt.plot(x,df.iloc[:, 8], marker='d',markersize=20,color="purple", label=df.keys()[8], linestyle="--",linewidth=5 )
# plt.plot(x,only_fine_denoising, marker='*',markersize=30, color="orange", label="Fine-denoising", linestyle="--",linewidth=5 )
# plt.plot(x,LC_GCN, marker='o', markersize=30,color="red", label="TSD",linestyle="--", linewidth=5)

# plt.legend(loc=0, numpoints=1)
plt.legend(loc=3, numpoints=1,frameon = False,labelspacing=2, fontsize=5)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=40,weight='bold')  # 设置图例字体的大小和粗细
# plt.xlabel("Number of perturbations per Nodes", fontdict=dict(fontsize=30, family='Times New Roman'))
# plt.ylabel("Accuracy of Attacked Nodes(%)", fontdict=dict(fontsize=30, family='Times New Roman'))
plt.xlabel("perturbation rate", fontdict=dict(fontsize=50, family='Times New Roman'),weight='bold')
plt.ylabel("Test Accuracy(%)", fontdict=dict(fontsize=50, family='Times New Roman'),weight='bold')

plt.xticks(fontsize=35,weight='bold')
plt.yticks(fontsize=35,weight='bold')
# plt.title('Citeseer',fontsize = 50,y=0.9,weight='bold')


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('./img/'+name+'.pdf', format='pdf')
plt.show()


