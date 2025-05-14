import json
import httpx
import openai
import re
import numpy as np

openai.api_key = ""

def extract_first_integer(message):
    match = re.search(r"\b([1-9]|10)\b", message)
    return int(match.group()) if match else None

def evaluate_explanation(head, relation, tail, logic_paths):
    path_section = "Explanation paths:\n"
    for i, path in enumerate(logic_paths, 1):
        path_section += f"{i}. {path}\n"

    prompt = f"""
    You are a link prediction evaluator. Please evaluate whether the following paths provide plausible explanations for the target triple, as a human expert would.

    Notice:
    A good plausible explanation should provide a clear logical and potentially causal connection to the target triple.
    Explanations based only on thematic similarity, shared patterns , or surface-level associations without strong reasoning should be rated lower.

    Target triple to explain:
    - Head entity: {head} 
    - Relation: {relation}
    - Tail entity: {tail}

    Explanations:
    {path_section}

    Relations may begin with `inverse_`, which indicates the reverse direction of the original relation.
    - Y -> inverse_has_part -> X means “Y is a part of X”

    Please answer the following questions for each path:
    1. Provide a plausibility score from 1 to 10 (10 = very plausible).
    2. Provide a brief justification, including the reason of the giving score.

    Please strictly follow this output format without any extra words:
    1. score: <int>, justification: "<your justification>"
    2. score: <int>, justification: "<your justification>"
    ... and so on
    """.strip()

    try:
        client = openai.OpenAI(
            base_url="",
            api_key=openai.api_key,
            http_client=httpx.Client(base_url="", follow_redirects=True),
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledge graph reasoning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        message = response.choices[0].message.content

        results = []
        for i, path in enumerate(logic_paths, 1):
            pattern = rf"{i}\.\s*score\s*[:：]?\s*(\d+).*?justification\s*[:：]?\s*['\"]?(.*?)['\"]?\s*(?=\n\d|$)"
            match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
            if match:
                score = int(match.group(1))
                justification = match.group(2).strip()
            else:
                score = None
                justification = "Not found"

            results.append({
                "path": path,
                "score": score,
                "justification": justification,
                "raw_response": message if i == 1 else "",
                "prompt": prompt if i == 1 else ""
            })

        return results

    except Exception as e:
        return [{
            "path": path,
            "score": None,
            "justification": str(e),
            "raw_response": "",
            "prompt": prompt
        } for path in logic_paths]

def extract_entity_glosses(target_triple, path_triples):
    entities = {}

    def add_entity(full_str):
        if ': ' in full_str:
            name, gloss = full_str.split(': ', 1)
            name = clean_entity_name(name.strip())
            gloss = gloss.strip()
            entities[name] = gloss

    parts = target_triple.split('\t')
    if len(parts) == 3:
        add_entity(parts[0])
        add_entity(parts[2])

    for triple_str in path_triples:
        parts = triple_str.split('\t')
        if len(parts) == 3:
            add_entity(parts[0])
            add_entity(parts[2])

    return entities

def clean_entity_name(name):
    return name.split('.')[0] if '.' in name else name

def parse_triple(triple_str):
    parts = triple_str.strip().split('\t')
    if len(parts) != 3:
        return None

    head_full, relation, tail_full = parts
    head, head_gloss = head_full.split(': ', 1)
    tail, tail_gloss = tail_full.split(': ', 1)
    return (head.strip().split('.')[0], head_gloss.strip(), relation.strip(), tail.strip().split('.')[0], tail_gloss.strip())

def build_chain_path(target_triple, path_triples):
    path_sequence = []

    def parse_entity(s):
        return clean_entity_name(s.split(':', 1)[0].strip())

    def parse_triple(triple_str):
        parts = triple_str.strip().split('\t')
        if len(parts) != 3:
            return None
        h = parse_entity(parts[0])
        r = parts[1].strip()
        t = parse_entity(parts[2])
        return (h, r, t)

    parsed_triples = [parse_triple(triple) for triple in path_triples]
    if any(p is None for p in parsed_triples):
        return "Invalid path"

    chain = []
    current_entity = None

    for i, (h, r, t) in enumerate(parsed_triples):
        if i == 0:
            chain.append(target_triple.split('\t')[0].split('.')[0])

            if h == chain[-1]:
                chain.extend([f"{r}", t])
                current_entity = t
            else:
                chain.extend([f"inverse_{r}", h])
                current_entity = h
        else:
            if h == current_entity:
                chain.extend([r, t])
                current_entity = t
            elif t == current_entity:
                chain.extend([f"inverse_{r}", h])
                current_entity = h
            else:
                chain.extend([r, t])
                current_entity = t

    return " -> ".join(chain)

def parse_json_to_data(input_json):
    data = []
    for triple, paths in input_json.items():
        head, relation, tail = triple.strip().split("\t")
        for path in paths:
            logical_path = build_chain_path(triple, path)
            data.append((head, relation, tail, logical_path))
    return data

def run_evaluation(data, json_data):
    results = {}
    grouped = {}

    for h, r, t, path in data:
        key = (h, r, t)
        grouped.setdefault(key, []).append(path)

    for i, ((h, r, t), paths) in enumerate(grouped.items()):
        print(f"[{i}] Evaluating: ({h.split('.')[0]}, {r}, {t.split('.')[0]})")
        path_results = evaluate_explanation(h.split('.')[0], r, t.split('.')[0], paths)
        results[(h.split('.')[0], r, t.split('.')[0])] = path_results

    return results

def save_results_to_file(results, filename):
    converted = {f"{h}\t{r}\t{t}": paths for (h, r, t), paths in results.items()}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)

def precision_at_k(llm_scores, k, threshold=7):
    top_k_scores = llm_scores[:k]
    positives = [s for s in top_k_scores if s is not None and s >= threshold]
    return len(positives) / k if k > 0 else 0

def average_score_at_k(llm_scores, k):
    top_k_scores = [s for s in llm_scores[:k] if s is not None]
    if not top_k_scores:
        return 0.0
    return sum(top_k_scores) / len(top_k_scores)

def dcg(scores):
    return sum((2 ** s - 1) / np.log2(i + 2) for i, s in enumerate(scores))

def ndcg_at_k(model_scores, llm_scores, k):
    if len(model_scores) < k:
        k = len(model_scores)

    order = np.argsort(model_scores)[::-1]
    ranked_true_scores = [llm_scores[i] for i in order[:k] if llm_scores[i] is not None]
    ideal_scores = sorted([s for s in llm_scores if s is not None], reverse=True)[:k]

    if not ranked_true_scores or not ideal_scores:
        return 0.0

    return dcg(ranked_true_scores) / dcg(ideal_scores)

def calculate_metrics(results, k1, k2, k3, threshold=5):
    p_at_k_list = []
    p_at_k_list1 = []
    ndcg_list = []
    avg_score_list = []
    for triple_key, path_infos in results.items():
        model_scores = list(range(len(path_infos), 0, -1))
        llm_scores = [p["score"] for p in path_infos]

        p_k = precision_at_k(llm_scores, k2, threshold)
        p_k1 = precision_at_k(llm_scores, k3, threshold)
        avg_k = average_score_at_k(llm_scores, k1)
        n_k = ndcg_at_k(model_scores, llm_scores, k1)
        p_at_k_list.append(p_k)
        p_at_k_list1.append(p_k1)
        avg_score_list.append(avg_k)
        ndcg_list.append(n_k)

    return {
        f"Precision@{k2}": round(np.mean(p_at_k_list), 4),
        f"Precision@{k3}": round(np.mean(p_at_k_list1), 4),
        f"AvgScore@{k1}": round(np.mean(avg_score_list), 4),
        f"nDCG@{k1}": round(np.mean(ndcg_list), 4),
        "Num Triples": len(results)
    }

if __name__ == "__main__":
    with open("baseline/ours/fb_transe_ab1_sem.json", "r") as f:
        input_json = json.load(f)

    data = parse_json_to_data(input_json)
    print(f"Loaded {len(data)} paths...")

    print("Evaluating explanations with LLM...")
    results = run_evaluation(data, input_json)

    save_results_to_file(results, "baseline/ours/fb_transe_ab1_results.json")

    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(results, k1=5, k2=1, k3=2, threshold=7)
    from pprint import pprint
    pprint(metrics)
