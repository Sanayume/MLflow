import numpy as np
import argparse
import time

# =============================================================================
# ❖❖❖ SCRIPT SOUL - MY SIGNATURE ❖❖❖
# (一个没有个性签名的脚本和咸鱼有什么区别喵？) O_o??
# (人类的本质是什么呢？当然写屎一样的代码喵！)
# =============================================================================
HEADER = r"""
        >_<
        (O o)
    /--( . . )--\
   |      U      |
   \  - . ' . -  /
    `'-.....-'`
+------------------------------------------------------------------+
|                                                                  |
|   Hello, World! 这里是Sana的个人空间喵~:                           |
|   >> Blog: https://sana.icu                                      |
|                                                                  |
|   如果这段代码让你会心一笑，请考虑给我的GitHub仓库点个Star喵!!       |
|   >> GitHub: [https://github.com/Sanayume/rainfalldata_research] |
|   >> GitHub: [https://github.com/Sanayume/MLflow]                |
+-------------------------------------------------------------------+
| 反向传播算法 - 禁咒咏唱版 >
"""

# =============================================================================
# ❖❖❖ 命令行启动器 - 每个合格的魔法师都应该有一个自定义的法杖 ❖❖❖
# =============================================================================
def setup_args():
    parser = argparse.ArgumentParser(
        description="神经网络の禁咒咏唱手册 - v1.0",
        epilog="詠唱開始: python %(prog)s --maou-mode",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--sasayaki', # 日语: ささやき (私语)
        action='store_true',
        help="[低语模式] 发动禁咒，窥探计算的深渊，展示每一步的魔力流动。"
    )
    parser.add_argument(
        '--mana',
        type=float,
        default=0.9,
        help="[魔力值] 设定学习率(η)，即你愿意为每次学习支付多少Mana。默认为0.9喵。"
    )
    parser.add_argument(
        '--maou-mode', # 日语: 魔王 (Maō)
        action='store_true',
        help="[魔王模式] 觉醒终极形态。没什么卵用，但气势上不能输。"
    )
    return parser.parse_args()

# =============================================================================
# ❖❖❖ 核心术式 - 这里是魔法发生的地方喵~ ❖❖❖
# =============================================================================

# 激活函数 • Sigmoid结界, 领域展开
# 领域效果：将无限的魔力洪流约束在0和1之间，赋予其意义。
def sigmoid_kekkai(z):
    # 召唤了NumPy上古魔神之力来计算欧拉数，保持克制。
    return 1.0 / (1.0 + np.exp(-z))

# Sigmoid结界的导数 • 因果律的残响
# 作用：计算当结界变动时，对世界造成的影响。
def sigmoid_prime(output):
    return output * (1.0 - output)

def shinkei_kakusei(args): # 日语: 神経覚醒 (神经觉醒)
    """主函数，即我们的“神经觉醒仪式”"""
    print(HEADER)
    if args.maou_mode:
        print("\n[警告] 检测到巨大的魔力波动...魔王模式已激活！「我が名はめぐみん！」\n")

    # --- 仪式准备：设定初始参数 ---
    print("「异世界传送门开启，正在同步初始参数...」")
    
    # 试炼之门
    shiren_no_mon = np.array([1, 0, 1])
    # 神之启示
    kami_no_kotae = 1
    # 魔力值
    mana = args.mana
    # 魔法书字典 • 契约书
    keiyaku_sho = {
        '14': 0.2, '15': -0.3, '24': 0.4, '25': 0.1,
        '34': -0.5, '35': 0.2, '46': -0.3, '56': -0.2
    }

    print(f"[WORLD INFO] 试炼之门........... {shiren_no_mon}")
    print(f"[WORLD INFO] 神之启示........... {kami_no_kotae}")
    print(f"[WORLD INFO] 初始Mana........... {mana}")
    print("-" * 60)

    # --- 咏唱第一章：圣なる計算---
    print("\n▷ ステップ１：聖なる計算（フォワードパス）")
    time.sleep(0.5)
    
    # 隐藏层魔力汇聚
    net_h4 = keiyaku_sho['14']*shiren_no_mon[0] + keiyaku_sho['24']*shiren_no_mon[1] + keiyaku_sho['34']*shiren_no_mon[2]
    out_h4 = sigmoid_kekkai(net_h4)
    net_h5 = keiyaku_sho['15']*shiren_no_mon[0] + keiyaku_sho['25']*shiren_no_mon[1] + keiyaku_sho['35']*shiren_no_mon[2]
    out_h5 = sigmoid_kekkai(net_h5)

    if args.sasayaki:
        print(f"  [低语] 神经元[4]的灵魂正在共鸣... 接收魔力 {net_h4:.4f}, 激活后输出 {out_h4:.4f}")
        print(f"  [低语] 神经元[5]的灵魂正在共鸣... 接收魔力 {net_h5:.4f}, 激活后输出 {out_h5:.4f}")

    # 输出层最终神谕
    net_o6 = keiyaku_sho['46']*out_h4 + keiyaku_sho['56']*out_h5
    final_output = sigmoid_kekkai(net_o6)
    print(f"「命运石之门的选择是...」 -> {final_output:.4f}")
    print("-" * 60)

    # --- 咏唱第二章：因果の逆流 (因果的逆流 / 反向传播) ---
    print("\n▷ ステップ２：因果の逆流（バックプロパゲーション）")
    time.sleep(0.5)

    # 计算最终的“世界线偏离度”
    error = kami_no_kotae - final_output
    # 输出层的因果律 (δ_o6)
    delta_o6 = error * sigmoid_prime(final_output)
    print(f"  [因果观测] 输出层因果律(δ_o6): {delta_o6:.4f}")

    # 隐藏层开始分锅，啊不，是追溯因果
    delta_h4 = (delta_o6 * keiyaku_sho['46']) * sigmoid_prime(out_h4)
    delta_h5 = (delta_o6 * keiyaku_sho['56']) * sigmoid_prime(out_h5)
    if args.sasayaki:
        print(f"  [低语] 神经元[4]承接的因果(δ_h4): {delta_h4:.4f}")
        print(f"  [低语] 神经元[5]承接的因果(δ_h5): {delta_h5:.4f}")
    print("-" * 60)

    # --- 咏唱第三章：契約更新 (契约更新 / 权重调整) ---
    print("\n▷ ステップ３：契約更新（ウェイトアップデート）")
    time.sleep(0.5)
    
    atarashii_keiyaku = {} # 新的契约书

    # 更新输入层到隐藏层的契约
    atarashii_keiyaku['14'] = keiyaku_sho['14'] + mana * delta_h4 * shiren_no_mon[0]
    atarashii_keiyaku['24'] = keiyaku_sho['24'] + mana * delta_h4 * shiren_no_mon[1] # 这位更是重量级，输入为0，摸鱼成功！
    atarashii_keiyaku['34'] = keiyaku_sho['34'] + mana * delta_h4 * shiren_no_mon[2]
    atarashii_keiyaku['15'] = keiyaku_sho['15'] + mana * delta_h5 * shiren_no_mon[0]
    atarashii_keiyaku['25'] = keiyaku_sho['25'] + mana * delta_h5 * shiren_no_mon[1] # 这位也摸了
    atarashii_keiyaku['35'] = keiyaku_sho['35'] + mana * delta_h5 * shiren_no_mon[2]
    
    # 更新隐藏层到输出层的契约
    atarashii_keiyaku['46'] = keiyaku_sho['46'] + mana * delta_o6 * out_h4
    atarashii_keiyaku['56'] = keiyaku_sho['56'] + mana * delta_o6 * out_h5

    # --- 最终章：契约重铸 ---
    print("\n「契约已重铸，世界线开始收束...」\n")
    print("=====================『第一次接触』任务报告=====================")
    print("  契约ID      旧版契约          新版契约          差异")
    print("-------------------------------------------------------------")
    for key in sorted(keiyaku_sho.keys()):
        old_val = keiyaku_sho[key]
        new_val = atarashii_keiyaku[key]
        delta = new_val - old_val
        print(f"  W{key:<5s}  {old_val:>12.5f}    ->   {new_val:>12.5f}    ({delta:+.5f})")
    print("=============================================================")
    print("\n世界线收束完毕")

if __name__ == '__main__':
    # 万物皆有起源，我们的故事从这里开始。
    # 不写main等于白给。
    args = setup_args()
    shinkei_kakusei(args)
    #杂鱼杂鱼, 在Linux下得用python3运行哦
    #在Windows下得用python运行哦
    #什么?macos?那就用python3运行吧
    #请给我点个star喵~
    #请给我点个star喵~