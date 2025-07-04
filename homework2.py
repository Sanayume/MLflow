import matplotlib.pyplot as plt
import numpy as np

# --- 配置 ---
xuehao_str = "2023218123"  # 学号字符串
banji_hao = 4  # 班级号（你在4班）

# --- Google配色（十六进制） ---
google_lan = '#4285F4'
google_hong = '#DB4437'
google_huang = '#F4B400'
google_lv = '#0F9D58'
google_cheng = '#FF6D00'  # 常用Google橙色

aixin_yanse_list = [google_hong, google_huang, google_lv, google_cheng, google_hong]  # 第5个再用红色

t_aixin = np.linspace(0, 2 * np.pi, 1000)
x_aixin = 16 * np.sin(t_aixin)**3
y_aixin = 13 * np.cos(t_aixin) - 5 * np.cos(2 * t_aixin) - 2 * np.cos(3 * t_aixin) - np.cos(4 * t_aixin)

xuehao_weishu_list = [int(d) for d in xuehao_str]

#风格
plt.style.use('seaborn-v0_8-whitegrid')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#初始化一个画布对象
fig = plt.figure(figsize=(18, 10))

#索引
aixin_yanse_idx = 0 

for i in range(1, 7): 
    ax = fig.add_subplot(2, 3, i) 

    if i == banji_hao:
        ax.plot(range(len(xuehao_weishu_list)), xuehao_weishu_list,
                marker='o', markersize=8, linestyle='-', linewidth=2.5, color=google_lan,
                label=f"学号: {xuehao_str}")
        ax.set_title(f"学号: {xuehao_str}", fontsize=14, color=google_lan, fontweight='bold')
        ax.set_xlabel("位序", fontsize=12)
        ax.set_ylabel("数字", fontsize=12)
        ax.set_xticks(range(len(xuehao_weishu_list)))
        ax.set_yticks(range(0, 10))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        dangqian_aixin_yanse = aixin_yanse_list[aixin_yanse_idx % len(aixin_yanse_list)]
        ax.plot(x_aixin, y_aixin, color=dangqian_aixin_yanse, linewidth=2.5)
        ax.set_title(f"爱心 {i}", fontsize=14, color=dangqian_aixin_yanse)
        ax.axis('on')
        ax.tick_params(axis='both', which='major', labelsize=10)
        aixin_yanse_idx += 1

plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])

#总标题
plt.suptitle(f"《大数据与人工智能》上机作业 - 学号: {xuehao_str} (班级: {banji_hao})",
             fontsize=20, fontweight='bold', color='#333333')

plt.savefig(f"上机作业_{xuehao_str}_Class{banji_hao}.png", dpi=300)
# plt.savefig(f"上机作业_{xuehao_str}_Class{banji_hao}_GoogleColors.pdf")

plt.show()

print(f"学号: {xuehao_str}")
print(f"学号各位数字: {xuehao_weishu_list}")
print(f"学号变化曲线在第{banji_hao}个子图: {banji_hao}")
print(f"爱心子图颜色依次为: {aixin_yanse_list}")
