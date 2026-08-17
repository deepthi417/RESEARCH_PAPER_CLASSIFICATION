"""
Phase 5 — Recommendation Engine. Deterministic KB queries only, no LLM.
Logic is unchanged from the validated Phase 5 implementation; only the
KB file paths are adjusted to this project's layout.
"""

import json
import os

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_base')


def _load_kb():
    with open(os.path.join(KB_DIR, 'technologies.json')) as f:
        technologies = json.load(f)
    with open(os.path.join(KB_DIR, 'algorithms.json')) as f:
        algorithms = json.load(f)
    with open(os.path.join(KB_DIR, 'metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(KB_DIR, 'resources.json')) as f:
        resources = json.load(f)
    return technologies, algorithms, metrics, resources


KB_COVERED_BROAD_DOMAINS = {
    'Artificial Intelligence', 'Theoretical Computer Science', 'Statistics',
    'Signal & Systems Engineering', 'Data & Information Systems',
    'Software & Programming', 'Systems & Networking', 'Scientific Computing',
}

DEEP_LEARNING_PROBLEM_TYPES = {
    'Image Classification', 'Object Detection', 'Generative Modeling',
    'Reinforcement Learning', 'Graph Analysis'
}


def _matches(entry_tags, target):
    if entry_tags == ['*'] or target in entry_tags:
        return True
    return False


def recommend_technologies(technologies, problem_type, data_type):
    results = []
    for key, tech in technologies.items():
        pt_match = _matches(tech['problem_types'], problem_type)
        dt_match = _matches(tech['data_types'], data_type)
        if pt_match and dt_match:
            score = 0
            score += 2 if problem_type in tech['problem_types'] else (1 if tech['problem_types'] == ['*'] else 0)
            score += 2 if data_type in tech['data_types'] else (1 if tech['data_types'] == ['*'] else 0)
            relevance = 'High' if score >= 3 else ('Medium' if score >= 1 else 'Low')
            results.append({
                'technology': tech['name'],
                'category': tech['category'],
                'relevance': relevance,
                'reason': tech['reason'],
                '_score': score
            })
    results.sort(key=lambda x: -x['_score'])
    for r in results:
        del r['_score']
    return results


def recommend_algorithms(algorithms, problem_type, data_type):
    results = []
    for key, algo in algorithms.items():
        if problem_type in algo['problem_types'] and (data_type in algo['data_types'] or algo['data_types'] == ['*']):
            results.append({
                'algorithm': algo['name'],
                'purpose': algo['purpose'],
                'reason': algo['reason']
            })
    return results


def recommend_metrics(metrics, problem_type):
    results = []
    for key, m in metrics.items():
        if problem_type in m['problem_types']:
            results.append({
                'metric': m['name'],
                'reason': m['reason']
            })
    return results


def recommend_resources(resources, problem_type, data_type, approach='deep_learning'):
    dataset_recs = list(resources['datasets']['generic'])
    for item in resources['datasets']['by_data_type'].get(data_type, []):
        if item not in dataset_recs:
            dataset_recs.append(item)

    hw = resources['hardware_rules'].get(approach, resources['hardware_rules']['classical_ml'])

    return {
        'datasets': dataset_recs,
        'software': resources['software'],
        'hardware': hw
    }


def estimate_difficulty(problem_type, data_type, broad_domain):
    score = 0
    reasons = []

    if problem_type in DEEP_LEARNING_PROBLEM_TYPES:
        score += 2
        reasons.append(f"'{problem_type}' typically requires deep learning models, raising implementation and compute complexity.")
    else:
        score += 1
        reasons.append(f"'{problem_type}' can often be approached with classical ML, which is comparatively simpler to implement.")

    if data_type == 'Image':
        score += 1
        reasons.append("Image data requires preprocessing pipelines (resizing, augmentation) and typically GPU access for practical training times.")
    elif data_type in ('Audio', 'Graph/Network'):
        score += 1
        reasons.append(f"{data_type} data requires specialized feature extraction not needed for simpler tabular/text data.")

    if broad_domain not in KB_COVERED_BROAD_DOMAINS:
        score += 1
        reasons.append(f"'{broad_domain}' is outside this tool's core knowledge-base coverage, requiring more independent research to fill recommendation gaps.")

    if score <= 2:
        level = 'Beginner'
    elif score == 3:
        level = 'Intermediate'
    elif score == 4:
        level = 'Advanced'
    else:
        level = 'Expert'

    return {'difficulty': level, 'score': score, 'reasons': reasons}


def full_recommendation(problem_type, data_type, broad_domain, subdomain):
    technologies, algorithms, metrics, resources = _load_kb()

    kb_covered = broad_domain in KB_COVERED_BROAD_DOMAINS

    tech_results = recommend_technologies(technologies, problem_type, data_type)
    algo_results = recommend_algorithms(algorithms, problem_type, data_type)
    metric_results = recommend_metrics(metrics, problem_type)

    approach = 'deep_learning' if problem_type in DEEP_LEARNING_PROBLEM_TYPES else 'classical_ml'
    resource_results = recommend_resources(resources, problem_type, data_type, approach)

    difficulty = estimate_difficulty(problem_type, data_type, broad_domain)

    return {
        'kb_coverage_warning': None if kb_covered else (
            f"'{broad_domain}' has limited knowledge-base coverage in this version — "
            f"the recommendations below are generic fallbacks, not domain-specific picks."
        ),
        'technologies': tech_results,
        'algorithms': algo_results,
        'metrics': metric_results,
        'resources': resource_results,
        'difficulty': difficulty
    }
