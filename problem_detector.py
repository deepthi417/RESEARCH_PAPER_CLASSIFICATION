"""
Phase 4 — rule-based problem-type / data-type detection.
Deterministic keyword matching, no ML model, no LLM.
"""

PROBLEM_TYPE_RULES = {
    'Image Classification': ['image classification', 'convolutional neural network', 'cnn', 'image recognition'],
    'Object Detection': ['object detection', 'bounding box', 'yolo', 'detection framework'],
    'Text Classification': ['text classification', 'document classification', 'sentiment analysis'],
    'Sequence Modeling': ['sequence to sequence', 'time series', 'lstm', 'recurrent neural network'],
    'Clustering': ['clustering', 'unsupervised', 'k-means', 'cluster analysis'],
    'Regression': ['regression', 'prediction of continuous', 'forecasting'],
    'Generative Modeling': ['generative adversarial', 'gan', 'diffusion model', 'generative model'],
    'Reinforcement Learning': ['reinforcement learning', 'policy gradient', 'q-learning', 'reward function'],
    'Optimization': ['optimization problem', 'convex optimization', 'gradient descent'],
    'Graph Analysis': ['graph neural network', 'graph analysis', 'network topology'],
}

DATA_TYPE_RULES = {
    'Image': ['image', 'images', 'pixel', 'visual', 'photograph'],
    'Text': ['text', 'document', 'corpus', 'language', 'sentence'],
    'Audio': ['audio', 'speech', 'acoustic', 'sound'],
    'Time Series': ['time series', 'temporal', 'sequential data'],
    'Graph/Network': ['graph', 'network structure', 'node', 'edge'],
    'Tabular': ['tabular', 'structured data', 'dataset of records'],
}


def detect_from_rules(text, rules):
    text_lower = text.lower()
    scores = {}
    for label, terms in rules.items():
        count = sum(text_lower.count(term) for term in terms)
        if count > 0:
            scores[label] = count
    if not scores:
        return 'Unclear / Not specified', {}
    best = max(scores, key=scores.get)
    return best, scores


def detect_problem_and_data_type(title, abstract):
    text = f"{title.strip()} {abstract.strip()}"
    problem_type, problem_scores = detect_from_rules(text, PROBLEM_TYPE_RULES)
    data_type, data_scores = detect_from_rules(text, DATA_TYPE_RULES)
    return problem_type, data_type
