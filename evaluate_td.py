import pandas as pd
import requests
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "tbiomed_results")

MODELS = {
    "ministral": "ministral-3:latest",
    "gemma": "gemma4:latest",
    "granite": "granite4.1:3b",
    "llama": "llama3.2:latest"
}

def llm_as_a_judge(prediction, ground_truth, model_name):
    prompt = f"""Are the concepts '{prediction}' and '{ground_truth}' semantically identical or synonyms? Reply strictly with only 'YES' or 'NO'. 
Examples: 
Concept A: 'cell phone' | Concept B: 'mobile phone' -> YES 
Concept A: 'lemon' | Concept B: 'citrus fruit' -> YES
Concept A: 'apple' | Concept B: 'bank' -> NO
Concept A: 'vehicle' | Concept B: 'truck' -> YES
Concept A: 'instrument' | Concept B: 'music' -> NO

Concept A: '{prediction}' | Concept B: '{ground_truth}' ->
Label:"""

    payload = {"model": model_name, "prompt": prompt, "stream": False}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=150)
            if res.status_code == 200:
                response_text = res.json().get("response", "").strip()
                print(f"   [LLM Judge] Ερώτηση: Είναι το '{prediction}' ίδιο με '{ground_truth}'; -> Απάντηση: '{response_text}'")
                return "YES" in response_text.upper()
        except Exception:
            if attempt < max_retries - 1:
                print(f"   [Καθυστέρηση] Το μοντέλο άργησε (προσπάθεια {attempt+1}/3). Ξανά σε 3 δευτερόλεπτα...")
                time.sleep(3)
            else:
                print("   [Σφάλμα] Το μοντέλο δεν απάντησε μετά από 3 προσπάθειες. Καταγράφεται ως Λάθος.")
                pass
    return False

def evaluate_predictions(model_suffix, model_name):
    predictions_file = os.path.join(RESULTS_DIR, f"predictions_{model_suffix}.csv")
    
    if not os.path.exists(predictions_file):
        print(f"Δεν βρέθηκε το αρχείο {predictions_file}. Τρέξε πρώτα το predict_td.py.")
        return

    print(f"\nΑξιολόγηση και Metrics (LLM Judge: {model_name})")
    preds_df = pd.read_csv(predictions_file)
    
    y_true = []
    y_pred = []

    for index, row in preds_df.iterrows():
        t_id = row['table_id']
        gt_word = str(row['Ground_Truth'])
        pred_word = str(row['LLM_Prediction'])
        
        is_correct = False
        if pred_word != "none_pred" and gt_word != "none_gt":
            if pred_word in gt_word or gt_word in pred_word or llm_as_a_judge(pred_word, gt_word, model_name):
                is_correct = True
                
        y_true.append(gt_word)
        
        if is_correct:
            y_pred.append(gt_word)
        else:
            y_pred.append(pred_word)
            
        print(f"[{t_id}] Πραγματικό: '{gt_word}' | Πρόβλεψη: '{pred_word}' -> {'Σωστό' if is_correct else 'Λάθος'}")

    correct_predictions = 0
    attempted_predictions = 0
    total_ground_truth = len(y_true)

    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            correct_predictions += 1
        
        if pred_val.lower() not in ["none_pred", "i don't know", "i don't know."]:
            attempted_predictions += 1

    precision = correct_predictions / attempted_predictions if attempted_predictions > 0 else 0.0
    
    recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    metrics_text = (
        f"\nTD Metrics (Μοντέλο: {model_name})\n"
        f"Precision: {precision:.2f}\n"
        f"Recall:    {recall:.2f}\n"
        f"F1-Score:  {f1:.2f}\n"
    )
    
    print(metrics_text)
    
    summary_file = os.path.join(RESULTS_DIR, "metrics_summary.txt")
    with open(summary_file, "a", encoding="utf-8-sig") as f:
        f.write(metrics_text)

if __name__ == "__main__":
    summary_file = os.path.join(RESULTS_DIR, "metrics_summary.txt")
    with open(summary_file, "w", encoding="utf-8-sig") as f:
        f.write("ΣΥΓΚΕΝΤΡΩΤΙΚΑ METRICS ΜΟΝΤΕΛΩΝ\n")
        
    for suffix, model in MODELS.items():
        evaluate_predictions(suffix, model)