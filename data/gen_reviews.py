"""生成模拟评论数据 data/reviews.json（确定性，重跑结果不变）。

    python data/gen_reviews.py

和商品库一样是模拟数据，用来验证“评论证据 -> 带引用的推荐理由”这条链路，不代表任何真实商品的口碑。
刻意构造了几类情况，评测和单测会用到：
- 卖点有好评支撑（大多数商品）；
- 卖点被评论反驳：商品标了“降噪”，但差评多于好评（E07、E12），推荐理由里不能把它当卖点；
- 好评差评都有的分歧项（每个商品随机一项）；
- 适合送礼 / 包装相关的评论（部分商品），对应“送人”这类场景需求。
评论里不写“X 元”这类价格表述：价格只能来自证据层，评论引用不能绕过价格事实校验。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 方面 -> (好评句式, 差评句式)
PHRASES: dict[str, tuple[list[str], list[str]]] = {
    "降噪": (["降噪效果明显，地铁上基本听不到报站", "开了降噪办公室空调声都没了", "飞机上戴着睡了一路，降噪很够用"],
           ["降噪效果一般，公交上还是很吵", "降噪开了有点压耳，效果也不明显"]),
    "长续航": (["续航很顶，一周充一次电", "连续用一整天还有电", "出差三天没带充电器也够用"],
             ["续航缩水，半天就要充", "电池掉电有点快"]),
    "轻量": (["很轻，戴久了也没感觉", "重量轻，拿在手里几乎没负担", "轻得像没穿一样，长距离不累"],
           ["没有宣传的那么轻", "分量偏重，久了有点累"]),
    "防水": (["下雨天跑步淋湿了也没事", "出汗多也不怕，防水靠谱"], ["进水后有杂音，防水存疑"]),
    "缓震": (["缓震脚感很软，跑十公里膝盖不疼", "落地很柔和，长距离跑很舒服"], ["缓震偏硬，跑久了脚掌疼"]),
    "热插拔": (["热插拔换轴很方便，新手也能玩", "支持热插拔，换了几种轴体都没问题"], ["热插拔座有点松，换轴时要小心"]),
    "5G": (["信号稳定，地下车库也有网", "5G 网速很快，下载大文件很爽"], ["信号偶尔掉格"]),
    "无线充电": (["无线充电很方便，放上就充", "无线充电速度挺快"], ["无线充电发热明显"]),
    "音质": (["音质清晰，低音有力度", "人声通透，听歌很舒服"], ["音质一般，高音有点刺"]),
    "佩戴": (["佩戴稳固，跑步不掉", "耳塞大小合适，戴久了不疼"], ["戴久了耳朵胀痛"]),
    "拍照": (["拍照清晰，夜景也不错", "人像模式很自然"], ["暗光拍照噪点多"]),
    "屏幕": (["屏幕显示细腻，看视频很舒服"], ["屏幕亮度在户外偏暗"]),
    "发热": (["打游戏也不怎么烫"], ["玩一会儿游戏就发烫"]),
    "透气": (["鞋面透气，夏天跑步不闷脚"], ["鞋面不透气，跑完很闷"]),
    "尺码": (["尺码标准，按平时码数买就行"], ["偏小半码，建议买大一码"]),
    "手感": (["敲击手感清脆，打字很舒服", "键程适中，手感扎实"], ["手感偏肉，不够清脆"]),
    "做工": (["做工扎实，没有异味", "细节处理得很好，没有毛刺"], ["做工粗糙，键帽有毛边"]),
    "送礼": (["包装很精致，送朋友很有面子", "买来送男朋友的，他很喜欢", "盒子很高级，直接当礼物送了"],
           ["包装太简陋，不适合送人"]),
    "性价比": (["这个价位很值", "性价比很高，同价位里最满意"], ["有点贵，同价位有更好的选择"]),
}

CATEGORY_ASPECTS = {
    "蓝牙耳机": ["音质", "佩戴"],
    "手机": ["拍照", "屏幕", "发热"],
    "跑鞋": ["透气", "尺码"],
    "机械键盘": ["手感", "做工"],
}

# 卖点被评论反驳的商品：差评多于好评
DISPUTED = {"E07": "降噪", "E12": "降噪"}
# 有送礼相关好评的商品
GIFT = {"E03", "E09", "E10", "P05", "P03", "S04", "K05", "K03"}


def h(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16)


def pick(options: list[str], *key: str) -> str:
    return options[h(*key) % len(options)]


def reviews_for(p: dict) -> list[dict]:
    pid, out = p["id"], []

    def add(aspect: str, positive: bool, n: int) -> None:
        pos, neg = PHRASES[aspect]
        options = pos if positive else neg
        for j in range(n):
            text = options[(h(pid, aspect) + j) % len(options)]  # 同一商品同一方面不重复
            rating = 5 if positive and h(pid, aspect, "r", str(j)) % 3 else (4 if positive else 2)
            out.append({"aspect": aspect, "rating": rating, "text": text})

    for f in p["features"]:
        if DISPUTED.get(pid) == f:
            add(f, True, 1)
            add(f, False, 2)
        else:
            add(f, True, 2)
    for a in CATEGORY_ASPECTS[p["category"]]:
        add(a, True, 1)
    # 每个商品一项好坏都有的分歧
    mixed = pick(CATEGORY_ASPECTS[p["category"]] + ["性价比"], pid, "mixed")
    add(mixed, False, 1)
    if pid in GIFT:
        add("送礼", True, 2)
    else:
        add("性价比", True, 1)

    for n, r in enumerate(out, 1):
        r["id"] = f"R-{pid}-{n}"
        r["product_id"] = pid
        r["date"] = f"2026-09-{1 + h(pid, str(n)) % 28:02d}"
    return [{k: r[k] for k in ("id", "product_id", "rating", "aspect", "text", "date")} for r in out]


def main() -> None:
    products = json.loads((ROOT / "products.json").read_text(encoding="utf-8"))
    reviews = [r for p in products for r in reviews_for(p)]
    (ROOT / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(reviews)} reviews for {len(products)} products")


if __name__ == "__main__":
    main()
