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
        "user": "Here is the claim: 'Long-termchangesin the radiometric calibration of the DNB itself are well understood and corrected (43).' Here is the abstract: 'The Visible Infrared Imaging Radiometer Suite (VIIRS) is a passive scanning spectroradiometer on board the Suomi-NPP satellite. It has 22 spectral bands including the day/night band (DNB), which is a panchromatic reflective solar band (RSB) covering a wavelength range of 500–900 nm. Similar to other RSBs, the radiometric calibration of the DNB is in reference to the sunlight reflected from an onboard solar diffuser (SD). As an independent validation to the SD measurement, lunar calibration has been regularly scheduled at nearly constant lunar phase. In this paper, the lunar calibration strategies developed for RSB are extended to DNB. The on-orbit gain coefficient, or the so-called F factor, is derived for DNB low-gain stage (LGS) from lunar data for each lunar calibration event. Its on-orbit change is compared with the change of the LGS SD F factor. For more accurate comparison, the impact of the on-orbit relative spectral responses (RSR) change, caused by the wavelength-dependent degradation of the optical throughput of VIIRS telescope mirrors, must be considered. This impact is more significant for DNB than other RSBs because of its wider bandwidth, and the impact to the SD and lunar calibrations are different due to different scene spectra. Simulation results show a gradually increased deviation of 1% between SD and lunar trends since launch till now and 0.3% deviation since 2 April 2012 lunar calibration, when the lunar F factor was firstly calculated, till now. Taking this effect into account, the on-orbit changes of the SD F factor and lunar F factor agree with each other in less than 0.3%. Our results validate the stability of the DNB SD calibration while demonstrating how the on-orbit RSR change should be considered in the radiometric calibration and data usage.'",
        "assistant": "SUPPORT",
    },
    {
        "user": "Here is the claim: 'Moreover, passivation of perovskite films, especially within the bulk film, using long-chain polymers that might coordinate with Pb2+ sites has rarely been reported (44), likely due to the challenges in film fabrication in the presence of a high–molecular weight polymer.' Here is the abstract: 'It is well known that the surface trap states and electronic disorders in the solution-processed CH3NH3PbI3 perovskite film affect the solar cell performance significantly and moisture sensitivity of photoactive perovskite material limits its practical applications. Herein, we show the surface modification of a perovskite film with a solution-processable hydrophobic polymer (poly(4-vinylpyridine), PVP), which passivates the undercoordinated lead (Pb) atoms (on the surface of perovskite) by its pyridine Lewis base side chains and thereby eliminates surface-trap states and non-radiative recombination. Moreover, it acts as an electron barrier between the perovskite and hole-transport layer (HTL) to reduce interfacial charge recombination, which led to improvement in open-circuit voltage (Voc) by 120 to 160 mV whereas the standard cell fabricated in same conditions showed Voc as low as 0.9 V owing to dominating interfacial recombination processes. Consequently, the power conversion efficiency (PCE) increased by 3 to 5 % in the polymer-modified devices (PCE=15 %) with Voc more than 1.05 V and hysteresis-less J–V curves. Advantageously, hydrophobicity of the polymer chain was found to protect the perovskite surface from moisture and improved stability of the non-encapsulated cells, which retained their device performance up to 30 days of exposure to open atmosphere (50 % humidity).'",
        "assistant": "CONTRADICT",
    },
    {
        "user": "Here is the claim: 'Spin-polarized DFT calculations were carried out using the VASP package36,37.' Here is the abstract: 'We present ab initio quantum-mechanical molecular-dynamics calculations based on the calculation of the electronic ground state and of the Hellmann-Feynman forces in the local-density approximation at each molecular-dynamics step. This is possible using conjugate-gradient techniques for energy minimization, and predicting the wave functions for new ionic positions using subspace alignment. This approach avoids the instabilities inherent in quantum-mechanical molecular-dynamics calculations for metals based on the use of a fictitious Newtonian dynamics for the electronic degrees of freedom. This method gives perfect control of the adiabaticity and allows us to perform simulations over several picoseconds.'",
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

