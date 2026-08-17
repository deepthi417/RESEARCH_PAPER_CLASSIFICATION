"""
Phase 6 — Project Recommendation. Template-based composition from
Phase 4 + Phase 5 outputs. No LLM. No independent invention of
technologies/algorithms/resources/metrics — everything here is a
direct pass-through of what Phase 5 already computed.
"""

GENERIC_KEYWORDS_TO_SKIP = {
    'deep learning', 'machine learning', 'using', 'based', 'approach',
    'method', 'study', 'paper', 'model', 'analysis', 'using collection'
}

PROBLEM_TYPE_TITLE_SUFFIX = {
    'Image Classification': 'Classification System',
    'Object Detection': 'Detection System',
    'Text Classification': 'Classification Platform',
    'Sequence Modeling': 'Forecasting System',
    'Clustering': 'Clustering & Segmentation Tool',
    'Regression': 'Prediction Tool',
    'Generative Modeling': 'Generation System',
    'Reinforcement Learning': 'Decision-Making Agent',
    'Optimization': 'Optimization Tool',
    'Graph Analysis': 'Network Analysis Tool',
    'Unclear / Not specified': 'Analysis Tool',
}

DATA_TYPE_INPUT_DESC = {
    'Image': 'A set of user-provided images (e.g. uploaded photos or a labeled image dataset)',
    'Text': 'Raw text documents or short text entries (e.g. user-submitted text, documents, or reviews)',
    'Audio': 'Audio clips or recordings (e.g. uploaded sound files)',
    'Time Series': 'Sequential/time-stamped numerical data (e.g. a CSV of historical readings)',
    'Graph/Network': 'A graph/network structure (e.g. nodes and edges as an adjacency list or edge list)',
    'Tabular': 'Structured tabular data (e.g. a CSV or database table of records)',
}

PROBLEM_TYPE_OUTPUT_DESC = {
    'Image Classification': 'A predicted class label per image, with a confidence score',
    'Object Detection': 'Bounding boxes with class labels and confidence scores for detected objects',
    'Text Classification': 'A predicted category/label per text input, with a confidence score',
    'Sequence Modeling': 'A forecast or predicted sequence continuation, with an uncertainty estimate where applicable',
    'Clustering': 'Cluster assignments for each input, with a visual/summary breakdown of cluster characteristics',
    'Regression': 'A predicted continuous value per input record',
    'Generative Modeling': 'Newly generated content (e.g. images) conditioned on the input',
    'Reinforcement Learning': 'A learned policy/action recommendation for a given state',
    'Optimization': 'An optimized configuration/solution for the given constraints',
    'Graph Analysis': 'Structural insights (e.g. important nodes, communities) about the input graph',
    'Unclear / Not specified': 'Structured output appropriate to the (currently undetected) task type',
}

DEEP_LEARNING_PROBLEM_TYPES = {
    'Image Classification', 'Object Detection', 'Generative Modeling',
    'Reinforcement Learning', 'Graph Analysis'
}


def _pick_title_keywords(keywords, max_words=2):
    filtered = [k for k in keywords if k.lower() not in GENERIC_KEYWORDS_TO_SKIP]
    if not filtered:
        filtered = keywords
    return filtered[:max_words]


def _make_title(keywords, problem_type):
    picked = _pick_title_keywords(keywords)
    phrase = ' '.join(w.capitalize() for k in picked for w in k.split())
    suffix = PROBLEM_TYPE_TITLE_SUFFIX.get(problem_type, 'Analysis Tool')
    if not phrase:
        return f"AI-Based {suffix}"
    return f"AI-Based {phrase} {suffix}"


def _architecture_sketch(problem_type, data_type):
    approach = 'Deep Learning' if problem_type in DEEP_LEARNING_PROBLEM_TYPES else 'Classical ML'
    return (
        f"User Input ({data_type}) -> Preprocessing -> "
        f"{'Feature Extraction (learned, via neural network)' if approach == 'Deep Learning' else 'Feature Engineering (manual/statistical)'} "
        f"-> {approach} Model -> Post-processing -> Output Delivery"
    )


def generate_project(title, abstract, broad_domain, subdomain, problem_type, data_type, keywords, phase5_result):
    warnings = []
    if phase5_result.get('kb_coverage_warning'):
        warnings.append(phase5_result['kb_coverage_warning'])

    if problem_type == 'Unclear / Not specified' or not phase5_result['technologies']:
        warnings.append(
            "Problem type could not be confidently detected, or the knowledge base returned no "
            "matching technologies for this combination — the project below is a generic template, "
            "not a specifically-tailored recommendation."
        )

    project_title = _make_title(keywords, problem_type)
    title_differs = project_title.strip().lower() != title.strip().lower()

    problem_statement = (
        f"Manually performing {problem_type.lower() if problem_type != 'Unclear / Not specified' else 'the relevant analysis'} "
        f"on {data_type.lower()} data in the {subdomain} space is time-consuming, inconsistent, and "
        f"does not scale to real-world volumes. There is a need for an automated, reusable system "
        f"that generalizes the class of technique described in the source research beyond its "
        f"original single-paper scope."
    )

    objective = (
        f"Design and build a {('deep learning' if problem_type in DEEP_LEARNING_PROBLEM_TYPES else 'machine learning')}-based "
        f"system that performs {problem_type.lower() if problem_type != 'Unclear / Not specified' else 'the detected task'} "
        f"on {data_type.lower()} data, applying the same class of approach as the source research to a "
        f"broader, deployable, user-facing tool rather than a single-dataset research result."
    )

    top_tech = [t['technology'] for t in phase5_result['technologies'][:4]]
    top_algo = [a['algorithm'] for a in phase5_result['algorithms'][:3]]
    proposed_solution = (
        f"Build a pipeline using {', '.join(top_tech) if top_tech else 'general-purpose ML tooling'} "
        f"for data handling and modeling, applying {', '.join(top_algo) if top_algo else 'an appropriate baseline algorithm'} "
        f"as the core modeling approach, wrapped in a simple interface for real users to submit new "
        f"{data_type.lower()} inputs and receive predictions."
    )

    key_features = [
        f"Automated {problem_type.lower() if problem_type != 'Unclear / Not specified' else 'analysis'} pipeline from raw {data_type.lower()} input to structured output",
        f"Interface for submitting new {data_type.lower()} data (upload/paste/batch-file, depending on interface choice)",
        f"Model evaluation dashboard reporting: {', '.join(m['metric'] for m in phase5_result['metrics'][:3]) if phase5_result['metrics'] else 'appropriate task metrics'}",
        "Clear display of prediction confidence, not just a bare label",
    ]

    future_enhancements = [
        "Expand training data coverage to additional related classes/categories",
        "Deploy as a hosted API endpoint for external integration",
        "Add model explainability (e.g. Grad-CAM for image models, SHAP for tabular models)"
        if problem_type in DEEP_LEARNING_PROBLEM_TYPES or data_type == 'Tabular'
        else "Add richer post-hoc explanation of predictions",
        "Compare against a stronger/larger model variant once a working baseline is validated",
    ]

    differs_from_source = (
        "The source research validates a specific technique on a specific dataset as a research "
        "contribution. This suggested project is a generalized, deployable application of the same "
        "class of technique — it targets a broader input distribution, adds a user-facing interface "
        "and evaluation dashboard, and is not a re-implementation of the paper's exact experiment."
    )

    why_suitable_parts = [
        f"The detected problem type ({problem_type}) and data type ({data_type}) map to a well-established "
        f"class of technique with known algorithms and evaluation metrics (see Recommended Algorithms/Metrics below)."
    ]
    if phase5_result.get('kb_coverage_warning'):
        why_suitable_parts.append(
            "Note: this domain has limited knowledge-base coverage in the current system version, so "
            "suitability here is based on generic ML applicability, not domain-specific validation."
        )
    else:
        why_suitable_parts.append(
            "This domain has established knowledge-base coverage in this system, so the technology and "
            "algorithm recommendations above are grounded in curated, domain-relevant entries rather than generic defaults."
        )

    return {
        'project_title': project_title,
        'problem_statement': problem_statement,
        'objective': objective,
        'proposed_solution': proposed_solution,
        'input': DATA_TYPE_INPUT_DESC.get(data_type, 'Structured input data appropriate to the detected data type'),
        'expected_output': PROBLEM_TYPE_OUTPUT_DESC.get(problem_type, 'Structured output appropriate to the detected task'),
        'recommended_architecture': _architecture_sketch(problem_type, data_type),
        'recommended_technologies': phase5_result['technologies'],
        'recommended_algorithms': phase5_result['algorithms'],
        'dataset_requirements': phase5_result['resources']['datasets'],
        'evaluation_metrics': phase5_result['metrics'],
        'difficulty': phase5_result['difficulty']['difficulty'],
        'estimated_development_complexity': phase5_result['difficulty']['reasons'],
        'key_features': key_features,
        'future_enhancements': future_enhancements,
        'why_suitable': ' '.join(why_suitable_parts),
        'differs_from_source': differs_from_source,
        'generation_warnings': warnings if warnings else None,
    }
