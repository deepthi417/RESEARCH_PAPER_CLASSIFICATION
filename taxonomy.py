"""
Phase 4 — Two-level taxonomy mapping (LOCKED, unchanged from original phase).
Pure lookup layer. Does not modify or reference the classifier.
"""

DOMAIN_MAP = {
    'cs.CV':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Computer Vision'},
    'cs.LG':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Machine Learning'},
    'cs.CL':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Natural Language Processing'},
    'cs.IT':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Information Theory'},
    'cs.RO':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Robotics'},
    'cs.CR':     {'broad_domain': 'Security & Privacy',             'subdomain': 'Cryptography and Security'},
    'cs.AI':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'AI - General'},
    'stat.ME':   {'broad_domain': 'Statistics',                     'subdomain': 'Methodology'},
    'stat.ML':   {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Machine Learning (Statistical)'},
    'cs.NI':     {'broad_domain': 'Systems & Networking',           'subdomain': 'Networking and Internet Architecture'},
    'cs.DS':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Data Structures and Algorithms'},
    'eess.SP':   {'broad_domain': 'Signal & Systems Engineering',   'subdomain': 'Signal Processing'},
    'eess.IV':   {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Image & Video Processing'},
    'cs.DC':     {'broad_domain': 'Systems & Networking',           'subdomain': 'Distributed Computing'},
    'eess.SY':   {'broad_domain': 'Signal & Systems Engineering',   'subdomain': 'Systems and Control'},
    'cs.SE':     {'broad_domain': 'Software & Programming',         'subdomain': 'Software Engineering'},
    'cs.LO':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Logic in Computer Science'},
    'cs.HC':     {'broad_domain': 'Human-Centered Computing',       'subdomain': 'Human-Computer Interaction'},
    'cs.SI':     {'broad_domain': 'Data & Information Systems',     'subdomain': 'Social and Information Networks'},
    'cs.CY':     {'broad_domain': 'Computing & Society',            'subdomain': 'Computers and Society'},
    'cs.IR':     {'broad_domain': 'Data & Information Systems',     'subdomain': 'Information Retrieval'},
    'stat.AP':   {'broad_domain': 'Statistics',                     'subdomain': 'Applications'},
    'cs.GT':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Computer Science and Game Theory'},
    'q-bio.PE':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Populations and Evolution'},
    'cs.NE':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Neural and Evolutionary Computing'},
    'cs.SD':     {'broad_domain': 'Signal & Systems Engineering',   'subdomain': 'Sound'},
    'q-bio.NC':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Neurons and Cognition'},
    'cs.DB':     {'broad_domain': 'Data & Information Systems',     'subdomain': 'Databases'},
    'eess.AS':   {'broad_domain': 'Signal & Systems Engineering',   'subdomain': 'Audio and Speech Processing'},
    'cs.SY':     {'broad_domain': 'Signal & Systems Engineering',   'subdomain': 'Systems and Control (CS)'},
    'cs.CC':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Computational Complexity'},
    'q-bio.QM':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Quantitative Methods'},
    'cs.PL':     {'broad_domain': 'Software & Programming',         'subdomain': 'Programming Languages'},
    'cs.DM':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Discrete Mathematics'},
    'cs.CG':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Computational Geometry'},
    'cs.CE':     {'broad_domain': 'Scientific Computing',           'subdomain': 'Computational Engineering, Finance & Science'},
    'stat.CO':   {'broad_domain': 'Statistics',                     'subdomain': 'Computation'},
    'cs.DL':     {'broad_domain': 'Data & Information Systems',     'subdomain': 'Digital Libraries'},
    'cs.AR':     {'broad_domain': 'Hardware & Architecture',        'subdomain': 'Hardware Architecture'},
    'cs.FL':     {'broad_domain': 'Theoretical Computer Science',   'subdomain': 'Formal Languages and Automata Theory'},
    'q-bio.BM':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Biomolecules'},
    'cs.MA':     {'broad_domain': 'Artificial Intelligence',        'subdomain': 'Multiagent Systems'},
    'cs.GR':     {'broad_domain': 'Software & Programming',         'subdomain': 'Computer Graphics'},
    'q-bio.MN':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Molecular Networks'},
    'cs.MM':     {'broad_domain': 'Signal & Systems Engineering',   'subdomain': 'Multimedia'},
    'cs.OH':     {'broad_domain': 'Computing & Computer Science',   'subdomain': 'Other Computer Science'},
    'cs.ET':     {'broad_domain': 'Systems & Networking',           'subdomain': 'Emerging Technologies'},
    'q-fin.ST':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Statistical Finance'},
    'q-bio.GN':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Genomics'},
    'q-fin.GN':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'General Finance'},
    'q-fin.MF':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Mathematical Finance'},
    'cs.SC':     {'broad_domain': 'Scientific Computing',           'subdomain': 'Symbolic Computation'},
    'q-fin.RM':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Risk Management'},
    'q-fin.PR':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Pricing of Securities'},
    'q-fin.CP':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Computational Finance'},
    'q-bio.TO':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Tissues and Organs'},
    'q-fin.PM':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Portfolio Management'},
    'cs.NA':     {'broad_domain': 'Scientific Computing',           'subdomain': 'Numerical Analysis'},
    'cs.PF':     {'broad_domain': 'Systems & Networking',           'subdomain': 'Performance'},
    'q-fin.TR':  {'broad_domain': 'Quantitative Finance',           'subdomain': 'Trading and Market Microstructure'},
    'cs.MS':     {'broad_domain': 'Scientific Computing',           'subdomain': 'Mathematical Software'},
    'q-bio.OT':  {'broad_domain': 'Quantitative Biology',           'subdomain': 'Other Quantitative Biology'},
    'stat.OT':   {'broad_domain': 'Statistics',                     'subdomain': 'Other Statistics'},
}

assert len(DOMAIN_MAP) == 63, f"Expected 63 classes, found {len(DOMAIN_MAP)}"


def get_domain_info(primary_category):
    mapping = DOMAIN_MAP.get(primary_category)
    if mapping:
        return mapping['broad_domain'], mapping['subdomain']
    return 'Unmapped', primary_category
