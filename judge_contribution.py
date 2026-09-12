import csv
import re
 
def normalize(text):
    t = text.strip().strip("'\".")
    t = t.lower()
    if t.startswith("label:"):
        t = t[len("label:"):].strip()
    t = t.replace("_", " ")
    t = re.sub(r"\([^)]*\)", "", t).strip()
    if " and " in t:
        t = t.split(" and ")[0].strip()
    return t
 
def is_error_or_idk(raw_pred):
    p = raw_pred.strip().lower()
    if p == "none_pred":
        return "none_pred"
    if "i don't know" in p or "i don t know" in p:
        return "idk"
    return None
 
def analyze(filepath):
    with open(filepath, encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))[1:]
    total = len(rows)
    none_pred = idk = correct = attempted = 0
    for row in rows:
        if len(row) < 3:
            continue
        gt_raw, pred_raw = row[1], row[2]
        status = is_error_or_idk(pred_raw)
        if status == "none_pred":
            none_pred += 1
            continue
        if status == "idk":
            idk += 1
            continue
        attempted += 1
        pred_norm, gt_norm = normalize(pred_raw), normalize(gt_raw)
        if pred_norm and gt_norm and (pred_norm in gt_norm or gt_norm in pred_norm):
            correct += 1
    return total, none_pred, idk, attempted, correct
 
FILES = {
    "Granite": "predictions_granite.csv",
    "Llama": "predictions_llama.csv",
    "Ministral": "predictions_ministral.csv",
    "Gemma": "predictions_gemma.csv",
}
 
FINAL_CORRECT = {"Granite": 308, "Llama": 146, "Ministral": 195, "Gemma": 259}
REPORTED = {
    "Granite":   (0.21, 0.19),
    "Gemma":     (0.18, 0.16),
    "Ministral": (0.17, 0.12),
    "Llama":     (0.10, 0.09),
}
 
print("Μοντέλο\tΣύνολο\tnone_pred\tIDK\tAttempted\tΛεξιλογική Ταύτιση\tP υπολ.\tR υπολ.\tP Πιν.5.2\tR Πιν.5.2")
for model, filepath in FILES.items():
    total, none_pred, idk, attempted, correct = analyze(filepath)
    final_correct = FINAL_CORRECT[model]
    p_calc = round(final_correct / attempted, 2) if attempted else 0.0
    r_calc = round(final_correct / total, 2) if total else 0.0
    p_rep, r_rep = REPORTED[model]
    print(f"{model}\t{total}\t{none_pred}\t{idk}\t{attempted}\t{correct}\t{p_calc}\t{r_calc}\t{p_rep}\t{r_rep}")