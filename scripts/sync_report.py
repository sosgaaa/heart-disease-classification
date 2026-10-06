"""Update marked tables and PGFPlots in report.tex from the saved evaluation JSON."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"baseline": "Majority baseline", "knn": "KNN", "kmeans": "K-Means", "logistic_regression": "Logistic regression"}
ORDER = tuple(NAMES)
SHORT_NAMES = {"baseline": "Baseline", "knn": "KNN", "kmeans": "K-Means", "logistic_regression": "LogReg"}


def blocks(result):
    config = result["config"]
    if (config["outer_folds"], config["inner_folds"], config["seed"]) != (5, 4, 42):
        raise ValueError("Update the report's protocol prose before using a different fold count or seed")
    if result["data"]["train_shape"] != [237, 13] or result["data"]["test_shape"] != [60, 13]:
        raise ValueError("The report describes the supplied 237/60 split with 13 features")
    methods = result["methods"]
    rows = [r"\begin{tabular}{@{}lrr@{}}\toprule",
            r"\multicolumn{3}{c}{Nested CV: mean $\pm$ SD}\\",
            r"Method & Accuracy (\%) & Macro F1\\\midrule"]
    for method in ORDER:
        record = methods[method]
        cv, test = record["nested_cv"], record["test"]
        rows.append(f"{SHORT_NAMES[method]} & ${cv['accuracy']['mean']:.2f} \\pm {cv['accuracy']['std']:.2f}$"
                    f" & ${cv['macro_f1']['mean']:.3f} \\pm {cv['macro_f1']['std']:.3f}$ \\\\")
    rows.extend([r"\midrule\multicolumn{3}{c}{Descriptive fixed test}\\",
                 r"Method & Accuracy (\%) & Macro F1\\\midrule"])
    for method in ORDER:
        test = methods[method]["test"]
        rows.append(f"{SHORT_NAMES[method]} & {test['accuracy']:.2f} & {test['macro_f1']:.3f} \\\\")
    rows.append(r"\bottomrule\end{tabular}")
    params = []
    for method in ORDER[1:]:
        p = methods[method]["selected_params"]
        if method == "logistic_regression":
            text = f"$\\eta={p['lr']:g}$, $T={p['max_iters']}$, $\\lambda={p['reg']:g}$"
        else:
            text = f"$k={p['k']}$, $f={p['features']}$"
        params.append(f"{SHORT_NAMES[method]} & {text} \\\\")
    best = max(ORDER[1:], key=lambda method: methods[method]["nested_cv"]["macro_f1"]["mean"])
    interpretation = (
        f"{NAMES[best]} has the highest mean nested-CV macro F1 in this run "
        f"({methods[best]['nested_cv']['macro_f1']['mean']:.3f}). "
        "The three mean scores are close relative to fold variation; this does not establish a reliable ranking. "
        "All three methods exceed the majority baseline on mean macro F1, but their absolute scores remain low."
    )
    plot_parts = []
    for method, title, xlabel in (("knn", "KNN", "Neighbours ($k$)"), ("kmeans", "K-Means", "Clusters ($k$)")):
        part = [r"\begin{minipage}{\linewidth}\centering", r"\begin{tikzpicture}",
                r"\begin{axis}[width=\linewidth,height=4cm,scale only axis=false,",
                f"title={{{title}}},xlabel={{{xlabel}}},ylabel={{Mean macro F1}},",
                r"ymin=0.1,ymax=0.45,grid=major,legend style={font=\scriptsize,at={(0.5,-0.4)},anchor=north,legend columns=3},",
                r"tick label style={font=\small},label style={font=\small},title style={font=\small}]" ]
        for feature_count, color, mark in ((7, "accent", "*"), (10, "orange!80!black", "square*"), (13, "green!50!black", "triangle*")):
            search = sorted((row for row in methods[method]["search"] if row["params"]["features"] == feature_count), key=lambda row: row["params"]["k"])
            coordinates = " ".join(f"({row['params']['k']},{row['cv']['macro_f1']['mean']:.6f})" for row in search)
            part.extend([f"\\addplot[color={color},mark={mark}] coordinates {{{coordinates}}};", f"\\addlegendentry{{$f={feature_count}$}}"])
        part.extend([r"\end{axis}\end{tikzpicture}\end{minipage}"])
        plot_parts.append("\n".join(part))
    classes = [r"\setlength{\tabcolsep}{2pt}",
               r"\begin{tabular}{@{}rr*{3}{r}@{}}\toprule",
               r"Class & Support & KNN & K-Means & LogReg\\\midrule"]
    for index, label in enumerate(result["data"]["classes"]):
        values = [str(label), str(result["data"]["test_class_counts"][index])]
        for method in ORDER[1:]:
            values.append("/".join(f"{methods[method]['test'][metric][index]:.3f}" for metric in ("class_recall", "class_f1")))
        classes.append(" & ".join(values) + r" \\")
    classes.append(r"\bottomrule\end{tabular}")
    matrices = []
    for method in ORDER[1:]:
        part = [r"\begin{minipage}{0.48\linewidth}\centering\small\setlength{\tabcolsep}{2pt}", f"\\textbf{{{SHORT_NAMES[method]}}}\\\\[4pt]",
                r"\begin{tabular}{@{}r*{5}{r}@{}}\toprule", r" & 0 & 1 & 2 & 3 & 4\\\midrule"]
        part += [str(index) + " & " + " & ".join(map(str, row)) + r" \\" for index, row in enumerate(methods[method]["test"]["confusion_matrix"])]
        part.append(r"\bottomrule\end{tabular}\end{minipage}")
        matrices.append("\n".join(part))
    return {"SUMMARY": "% Numerical content generated by scripts/sync_report.py; do not edit marked blocks by hand.",
            "RESULTS": "\n".join(rows), "PARAMETERS": "\n".join(params), "INTERPRETATION": interpretation,
            "PLOTS": "\n\\par\\medskip\n".join(plot_parts), "CLASSES": "\n".join(classes),
            "CONFUSION": matrices[0] + "\n\\hfill\n" + matrices[1] + "\n\\par\\medskip\n" + matrices[2]}


def render(source, result):
    for key, body in blocks(result).items():
        pattern = rf"(% BEGIN GENERATED {key}\n).*?(% END GENERATED {key})"
        source, count = re.subn(pattern, lambda match: match[1] + body + "\n" + match[2], source, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected exactly one {key} block, found {count}")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/evaluation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "report.tex")
    parser.add_argument("--check", action="store_true", help="fail if numerical content is out of sync")
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    if hashlib.sha256((ROOT / "features.npz").read_bytes()).hexdigest() != result["data"]["sha256"]:
        raise SystemExit("Dataset differs from the saved evaluation: rerun evaluate.py")
    for relative, checksum in result["provenance"]["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != checksum:
            raise SystemExit(f"Source differs from the saved evaluation ({relative}): rerun evaluate.py")
    source = args.report.read_text()
    updated = render(source, result)
    if args.check:
        if source != updated:
            raise SystemExit("Report is out of sync: run python scripts/sync_report.py")
        print("Report tables and plots match the saved evaluation.")
    else:
        args.report.write_text(updated)
        print(f"Updated {args.report}")


if __name__ == "__main__":
    main()
