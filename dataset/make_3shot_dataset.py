"""
将 dataset 文件夹中的现有 jsonl 数据（0-shot）转换为 3-shot 结构。

设计原则：
- **不修改原有提示词内容**：原有的 system / user / assistant 内容完全保留；
- 仅在原有对话的前面插入 3 个「示例轮次」；
- 你可以在本脚本顶部直接编辑 3-shot 的 user / assistant 内容。

转换后的结构（单条数据）大致为：
    [system,
     user_example_1,
     assistant_example_1,
     user_example_2,
     assistant_example_2,
     user_example_3,
     assistant_example_3,
     原来的 user,
     原来的 assistant]

使用方式：
    1. 先在下面的 FEW_SHOT_EXAMPLES 中，把三个示例的 user / assistant 文本改成你想要的内容；
    2. 在项目根目录执行：
           python -m dataset.make_3shot_dataset
    3. 会在 dataset 目录下为每个 *.jsonl 生成一个 *_3shot.jsonl 文件。
"""

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"


# 你可以在这里直接填写 3 个示例的内容
# 示例格式建议保持和现在数据集中 user / assistant 的风格一致
FEW_SHOT_EXAMPLES = [
    {
        "user": "Here is the claim: 'The high availability of methane as the principal component of natural gas, calls for the development of efficient methods for its conversion into commodity chemicals and liquid fuels (3–7).' Here is the abstract: 'The efficient use of natural gas will require catalysts that can activate the first C–H bond of methane while suppressing complete dehydrogenation and avoiding overoxidation. We report that single iron sites embedded in a silica matrix enable direct, nonoxidative conversion of methane, exclusively to ethylene and aromatics. The reaction is initiated by catalytic generation of methyl radicals, followed by a series of gas-phase reactions. The absence of adjacent iron sites prevents catalytic C-C coupling, further oligomerization, and hence, coke deposition. At 1363 kelvin, methane conversion reached a maximum at 48.1% and ethylene selectivity peaked at 48.4%, whereas the total hydrocarbon selectivity exceeded 99%, representing an atom-economical transformation process of methane. The lattice-confined single iron sites delivered stable performance, with no deactivation observed during a 60-hour test.'",
        "assistant": "SUPPORT",
    },
    {
        "user": "Here is the claim: 'For example, a recent study exploring the mechanisms of liver regeneration showed that IL-4 receptor signaling in hepatocytes is required for proliferation after injury, with eosinophils delivering the IL-4 (49).' Here is the abstract: 'The liver is a central organ for the synthesis and storage of nutrients,production of serum proteins and hormones, and breakdown oftoxins and metabolites. Because the liver is susceptible to toxin- orpathogen-mediated injury, it maintains a remarkable capacity toregenerate by compensatory growth. Speciﬁcally, in response toinjury, quiescent hepatocytes enter the cell cycle and undergo DNAreplication to promote liver regrowth. Despite the elucidation ofa number of regenerative factors, the mechanisms by which liverinjury triggers hepatocyte proliferation are incompletely understood.We demonstrate here that eosinophils stimulate liver regenerationafter partial hepatectomy and toxin-mediated injury. Liver injuryresults in rapid recruitment of eosinophils, which secrete IL-4 topromote the proliferation of quiescent hepatocytes. Surprisingly,signaling via the IL-4Rα in macrophages, which have been implicatedin tissue repair, is dispensable for hepatocyte proliferation and liverregrowth after injury. Instead, IL-4 exerts its proliferative actions viaIL- 4Rα in hepatocytes. Our ﬁndings thus provide a u ni que mech-anism by which eosi nophil-deri ved IL-4 stimulates hepatocyteproliferation in regenerating liver.'",
        "assistant": "CONTRADICT",
    },
    {
        "user": "Here is the claim: 'Different from other hydrogen-storage metals such as yttrium17, which are associated with poor optical response.' Here is the abstract: 'A key challenge for the development of active plasmonic nanodevices is the lack of materials with fully controllable plasmonic properties. In this work, we demonstrate that a plasmonic resonance in top-down nanofabricated yttrium antennas can be completely and reversibly turned on and off using hydrogen exposure. We fabricate arrays of yttrium nanorods and optically observe, in extinction spectra, the hydrogen-induced phase transition between the metallic yttrium dihydride and the insulating trihydride. Whereas the yttrium dihydride nanostructures exhibit a pronounced particle plasmon resonance, the transition to yttrium trihydride leads to a complete vanishing of the resonant behavior. The plasmonic resonance in the dihydride state can be tuned over a wide wavelength range by simply varying the size of the nanostructures. Furthermore, we develop an analytical diffusion model to explain the temporal behavior of the hydrogen loading and unloading trajectories observed in our experiments and gain information about the thermodynamics of our device. Thus, our nanorod system serves as a versatile basic building block for active plasmonic devices ranging from switchable perfect absorbers to active local heating control elements.'",
        "assistant": "NULL",
    },
]


def convert_file(path: Path) -> Path:
    """将单个 jsonl 文件转换为 3-shot 版本，返回新文件路径。"""
    target = path.with_name(path.stem + "_3shot" + path.suffix)

    with path.open("r", encoding="utf-8") as f_in, target.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])

            new_messages = []
            inserted_examples = False

            for msg in messages:
                # 在第一个非 system 消息之前插入 3 组 few-shot 示例
                if not inserted_examples and msg.get("role") != "system":
                    for ex in FEW_SHOT_EXAMPLES:
                        new_messages.append(
                            {
                                "role": "user",
                                "content": ex["user"],
                            }
                        )
                        new_messages.append(
                            {
                                "role": "assistant",
                                "content": ex["assistant"],
                            }
                        )
                    inserted_examples = True

                new_messages.append(msg)

            obj["messages"] = new_messages
            f_out.write(json.dumps(obj, ensure_ascii=False))
            f_out.write("\n")

    return target


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"未找到数据集目录：{DATASET_DIR}")

    jsonl_files = sorted(DATASET_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"{DATASET_DIR} 下未找到任何 jsonl 文件。")
        return

    print("开始转换以下文件为 3-shot 版本：")
    for path in jsonl_files:
        print(f"  - {path.name}")
        new_path = convert_file(path)
        print(f"    → 生成：{new_path.name}")

    print("全部完成。你现在可以在 *_3shot.jsonl 中检查 3 个示例是否正确插入。")


if __name__ == "__main__":
    main()

