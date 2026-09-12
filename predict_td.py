import pandas as pd
import requests
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "tbiomed_results")
TABLES_DIR = r"C:\Users\<user>\OneDrive\Documents\ptyxiaki\tbiomedical2-splits - without  test gt\horizontal\val\tables"
TD_FILE = r"C:\Users\<user>\OneDrive\Documents\ptyxiaki\tbiomedical2-splits - without  test gt\horizontal\val\gt\td_gt.csv"

MODELS = {
    "ministral": "ministral-3:latest",
    "gemma": "gemma4:latest",
    "granite": "granite4.1:3b",
    "llama": "llama3.2:latest"
}

wikidata_cache = {}

def translate_wikidata_uris(uris_string):
    labels = []
    for uri in str(uris_string).split(','):
        uri = uri.strip()
        if not uri or 'wikidata.org' not in uri: continue
        
        if uri in wikidata_cache:
            if wikidata_cache[uri]: labels.append(wikidata_cache[uri])
            continue

        entity_id = uri.rstrip("/").split("/")[-1].upper()
        api_url = "https://www.wikidata.org/w/api.php"
        params = {"action": "wbgetentities", "ids": entity_id, "format": "json", "props": "labels"}
        headers = {"User-Agent": "PtyxiakiLLM/1.0"}
        
        try:
            res = requests.get(api_url, params=params, headers=headers)
            if res.status_code == 200:
                data = res.json()
                label = data["entities"][entity_id]["labels"]["en"]["value"]
                labels.append(label.lower())
                wikidata_cache[uri] = label.lower()
            else:
                wikidata_cache[uri] = None
        except Exception:
            wikidata_cache[uri] = None
            
    return labels

def call_llm_for_td(table_df, model_name):
    clean_df = table_df.head(100).fillna("")
    table_text = clean_df.to_markdown(index=False)

    prompt = f"""Task: Solve the Topic Detection (TD) Task of SemTab Challenge, i.e. identify the specific core entity type or main topic of the provided table.

Rules:
1. Ignore empty cells.
2. Provide the formal, specific singular entity name as it would appear on wikidata or dbpedia.
3. Provide only the label. (No comments or explanation).
4. If you cannot determine the topic, reply strictly with 'i don't know'.

Example 1:
| col0 | col1 | col2 |
|---|---|---|
| Id1 | Toyota | Corolla |
| Id2 | Ford | Mustang |
| Id3 | Honda |  |
Label: car

Example 2:
| col0 | col1 | col2 |
|---|---|---|
| X1 | Apple | iPhone 14 |
| X2 | Samsung | Galaxy S23 |
Label: smartphone

Now, analyze the following table:

{table_text}

Label:"""

    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
        if res.status_code != 200: return "none_pred"

        response_text = res.json().get("response", "").strip().strip("'\".").lower()
        response_text = response_text.replace("label:", "").replace("_", " ").split("(")[0].strip().split(" and ")[0].strip()

        return response_text if response_text else "none_pred"
    except Exception as e: 
        print(f"Σφάλμα LLM: {e}")
        return "none_pred"

def generate_predictions(model_suffix, model_name, limit=None):
    predictions_file = os.path.join(RESULTS_DIR, f"predictions_{model_suffix}.csv")
    
    try:
        td_df = pd.read_csv(TD_FILE, header=None, names=['table_id', 'Label'])
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του αρχείου TD: {e}")
        return

    print(f"----------------------------------------------------\n")
    print(f"Προβλέψεις Μοντέλου: {model_name}")
    print(f"Αρχείο αποθήκευσης: {predictions_file}")
    print(f"----------------------------------------------------\n")
    
    if limit:
        td_df = td_df.sample(n=limit, random_state=42)
        print(f"Τρέχουμε {limit} τυχαίους πίνακες.\n")
        
    processed_ids = set()
    if os.path.exists(predictions_file):
        existing_df = pd.read_csv(predictions_file)
        processed_ids = set(existing_df['table_id'].astype(str).tolist())
        print(f"Βρέθηκαν {len(processed_ids)} ήδη αποθηκευμένες προβλέψεις. Συνεχίζουμε με τις υπόλοιπες.\n")

    for index, row in td_df.iterrows():
        t_id = str(row['table_id']).strip()
        
        if t_id in processed_ids:
            continue
            
        table_path = os.path.join(TABLES_DIR, f"{t_id}.csv")
        if not os.path.exists(table_path): continue

        gt_labels = translate_wikidata_uris(str(row['Label']))
        gt_word = gt_labels[0] if gt_labels else "none_gt"
        
        df = pd.read_csv(table_path)
        pred_word = call_llm_for_td(df, model_name)
        
        print(f"[{t_id}] Πραγματικό: '{gt_word}' | LLM: '{pred_word}'")
        
        new_row = pd.DataFrame([{
            "table_id": t_id,
            "Ground_Truth": gt_word,
            "LLM_Prediction": pred_word
        }])
        
        new_row.to_csv(predictions_file, mode='a', header=not os.path.exists(predictions_file), index=False)
        
        time.sleep(2)

    print(f"\nΗ διαδικασία ολοκληρώθηκε για το μοντέλο {model_name}.\n")

if __name__ == "__main__":
    for suffix, model in MODELS.items():
        generate_predictions(suffix, model, limit=None)