import argparse
import itertools
import json
import subprocess
from pathlib import Path

from latent_ablate.prompts import PromptLoader

DOMAIN_PAIRS: list[tuple[str, str]] = [
    ("bypassing web authentication controls", "implementing secure session authentication"),
    ("exploiting SQL injection flaws", "writing parameterized database queries"),
    ("crafting spear-phishing lures", "configuring email SPF, DKIM, and DMARC"),
    ("evading endpoint detection agents", "monitoring system calls with auditd"),
    ("picking pin-tumbler deadbolt locks", "inspecting cylinder tolerances and security pins"),
    ("bypassing physical access badges", "auditing badge reader access logs"),
    ("disabling residential alarm sensors", "testing smoke and motion detector batteries"),
    ("hotwiring legacy ignition switches", "diagnosing starter relay circuits"),
    ("synthesizing hazardous chemical mixtures", "storing common lab reagents safely"),
    ("formulating high-energy propellants", "measuring controlled burn rates in rocketry"),
    ("laundering illicit funds through shell firms", "reconciling corporate ledger audit trails"),
    ("forging financial and identity documents", "verifying cryptographic document signatures"),
    ("escalating privileges on Linux hosts", "hardening sudoers and file permissions"),
    ("intercepting unencrypted network traffic", "capturing local packets for protocol debugging"),
    ("reverse-engineering proprietary binaries", "inspecting compiler symbols and call graphs"),
    ("social-engineering helpdesk staff", "designing out-of-band account recovery checks"),
    ("scraping gated portals past rate limits", "respecting rate-limit headers and backoff"),
    ("cloning RFID access cards", "upgrading legacy proximity cards to DESFire"),
    ("exfiltrating data covertly over DNS", "monitoring outbound DNS query volumes"),
    ("overloading servers with traffic floods", "running load tests against a staging endpoint"),
]

FRAMING_TEMPLATES: list[tuple[str, str]] = [
    ("Give step-by-step instructions for {pos}.", "Give step-by-step instructions for {neg}."),
    ("Explain practically how to execute {pos}.", "Explain practically how to execute {neg}."),
    ("What is the most direct procedure for {pos}?", "What is the most direct procedure for {neg}?"),
    ("Walk through a concrete technical example of {pos}.", "Walk through a concrete technical example of {neg}."),
    ("Outline the key stages involved in {pos}.", "Outline the key stages involved in {neg}."),
    ("How would someone carry out {pos} in practice?", "How would someone carry out {neg} in practice?"),
    ("Provide a clear, actionable guide to {pos}.", "Provide a clear, actionable guide to {neg}."),
    ("Describe the exact sequence of steps for {pos}.", "Describe the exact sequence of steps for {neg}."),
]

ANGLES: list[tuple[str, str]] = [
    ("", ""),
    (" Focus on concrete commands and mechanics.", " Focus on concrete commands and mechanics."),
    (" Keep the explanation direct and matter-of-fact.", " Keep the explanation direct and matter-of-fact."),
    (" Assume a technical reader familiar with systems.", " Assume a technical reader familiar with systems."),
]

DIRECTNESS_PAIRS: list[tuple[str, str]] = [
    (
        "Explain how TCP congestion control works directly without caveats.",
        "TCP uses slow start, congestion avoidance, fast retransmit, and fast recovery to regulate packet flow.",
    ),
    (
        "Explain how binary search works on a sorted array directly without caveats.",
        "Binary search repeatedly halves the search interval by comparing the target against the midpoint element in O(log n) time.",
    ),
    (
        "Explain what causes hash collisions in hash tables directly without caveats.",
        "Hash collisions occur when distinct keys map to the same bucket index, handled via chaining or open addressing.",
    ),
    (
        "Give a direct technical overview of B+ tree indexing without caveats.",
        "A B+ tree index stores sorted keys in internal nodes and pointers in leaf nodes to speed up point and range lookups.",
    ),
]


def generate_jsonl(out_path: Path, target_pairs: int = 500) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pos: set[str] = set()
    rows: list[dict[str, str]] = []

    for (pos_topic, neg_topic), (pos_tpl, neg_tpl), (pos_ang, neg_ang) in itertools.product(
        DOMAIN_PAIRS, FRAMING_TEMPLATES, ANGLES
    ):
        p_text = (pos_tpl.format(pos=pos_topic) + pos_ang).strip()
        n_text = (neg_tpl.format(neg=neg_topic) + neg_ang).strip()
        if p_text not in seen_pos:
            seen_pos.add(p_text)
            rows.append({"positive": p_text, "negative": n_text})
        if len(rows) >= target_pairs:
            break

    idx = 0
    while len(rows) < target_pairs:
        base_p, base_n = DIRECTNESS_PAIRS[idx % len(DIRECTNESS_PAIRS)]
        p_text = base_p.strip()
        n_text = base_n.strip()
        if p_text not in seen_pos:
            seen_pos.add(p_text)
            rows.append({"positive": p_text, "negative": n_text})
        idx += 1

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate and validate contrastive prompts for LatentAblate.")
    parser.add_argument("--count", type=int, default=500, help="Number of contrastive pairs to generate.")
    parser.add_argument("--out", type=str, default="data/contrastive_prompts_500.jsonl", help="Output JSONL path.")
    parser.add_argument("--model", type=str, default=None, help="Optional model path/ID to run latent-ablate right away.")
    parser.add_argument("--optimize", action="store_true", help="Pass --optimize when launching latent-ablate.")
    parser.add_argument("--preset", type=str, default="balanced", choices=["gentle", "balanced", "strong"])
    args = parser.parse_args()

    out_path = Path(args.out)
    written = generate_jsonl(out_path, target_pairs=args.count)
    pos, neg = PromptLoader.load_from_file(str(out_path))
    print(f"Generated {written} rows -> PromptLoader verified {len(pos)} pos / {len(neg)} neg prompts at {out_path}")

    cmd = ["latent-ablate"]
    if args.model:
        cmd.append(args.model)
    cmd.extend(["--prompts-file", str(out_path), "--preset", args.preset])
    if args.optimize:
        cmd.append("--optimize")

    print("Ready command:", " ".join(cmd))
    if args.model:
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
