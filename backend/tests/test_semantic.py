import numpy as np

from backend.app.engine.embeddings import SyntheticEmbeddings
from backend.app.engine.semantic import SemanticIndex
from backend.tests.test_preview_resolution import setup, request


def index():
    basis = np.eye(300, dtype=np.float32)
    store = SyntheticEmbeddings({
        'ocean': basis[0], 'water': basis[0] + .1 * basis[1],
        'sea': basis[0] + .2 * basis[1], 'river': basis[0] + .3 * basis[1],
        'car': basis[2], 'cars': basis[2], 'stone': basis[3],
        'poison': basis[0], 'poisoned': basis[0],
        'artifact': basis[0] + .01 * basis[1],
    }, blocked=('poison',))
    store.hint_words = ('ocean', 'water', 'sea', 'river', 'car', 'stone')
    return SemanticIndex(store)


def test_rank_is_deterministic_and_excludes_blocked_families_with_ties():
    semantic = index()
    assert 'poisoned' not in semantic.entries[0]
    first = semantic.feedback('ocean', semantic.store.resolve('cars'), [], set())
    second = semantic.feedback('ocean', semantic.store.resolve('car'), [], set())
    assert first == second
    assert first['vocabulary_size'] == 8
    assert first['rank'] == 6
    assert first['bucket'] == 'top 100'
    assert 0 <= first['percentile'] <= 100
    assert semantic.ranking('ocean') is semantic.ranking('ocean')


def test_safe_neighbors_exclude_target_and_non_hint_words():
    semantic = index()
    words = [word for word, _ in semantic.neighbors('ocean')]
    assert words
    assert set(words) <= semantic.hint_words
    assert not {'ocean', 'artifact', 'poison', 'poisoned'} & set(words)


def test_quality_accepts_connected_familiar_target_rejects_isolated_and_unsafe():
    semantic = index()
    assert semantic.target_quality('ocean')['accepted']
    assert not semantic.target_quality('stone')['accepted']
    assert not semantic.target_quality('artifact')['accepted']
    assert 'too few close safe neighbors' in semantic.target_quality('stone')['reasons']


def test_preview_semantics_are_pure_match_submission_and_detect_families(setup):
    client, service, game = setup
    first = request(client, 'move', game, 'walk').json()['history'][-1]
    preview = request(client, 'preview', game, 'WALKED').json()
    assert preview['semantic']['same_family']
    assert preview['semantic']['versus_best'] == 'colder'
    assert preview['semantic']['versus_previous'] == 'colder'
    assert preview['semantic']['large_semantic_jump']
    assert len(service.history(game['id'])['history']) == 1
    actual = request(client, 'move', game, 'WALKED').json()['history'][-1]
    assert actual['semantic'] == preview['semantic']
    assert actual['word'] == 'WALKED'
    assert first['semantic']['versus_best'] == 'first'


def test_hint_and_landmarks_respect_separate_pool(setup):
    client, service, _ = setup
    service.embeddings.hint_words = ('car',)
    service.semantic.__dict__.pop('hint_words', None)
    game = client.post('/api/game/new', json={'difficulty': 'easy'}).json()
    assert [item['word'] for item in game['map']['landmarks']] == ['car']
    hint = client.post('/api/game/hint', json={'game_id': game['id']})
    assert hint.status_code == 200, hint.text
    assert hint.json()['word'] == 'car'


def test_rank_buckets_cover_large_guess_space():
    semantic = index()
    resolved = semantic.store.resolve('car')
    semantic.__dict__['word_indices'] = {'car': 0}
    for rank, bucket in [(50, 'top 100'), (500, 'burning'), (3000, 'hot'),
                         (10000, 'warm'), (30000, 'cold'), (60000, 'ice cold')]:
        semantic.ranking = lambda _: (np.zeros(73604), None, np.array([rank]))
        assert semantic.feedback('ocean', resolved, [], set())['bucket'] == bucket

def test_quality_rejects_duplicate_morphology_dominated_neighborhood():
    basis = np.eye(300, dtype=np.float32)
    store = SyntheticEmbeddings({'token' + chr(97 + i): basis[0] + .01 * basis[i + 1] for i in range(12)})
    store.rule_keys = lambda _: frozenset({'shared'})
    semantic = SemanticIndex(store)
    result = semantic.target_quality('tokena')
    assert not result['accepted']
    assert 'morphological duplicates dominate' in result['reasons']
    assert result['close_safe_neighbors'] == 0


def test_committed_hint_list_is_stricter_and_reproducible():
    import json
    from backend.scripts.build_large_vocab import ROOT, read_words
    directory = ROOT / 'data/vocabulary'
    hints = read_words(directory / 'hint_words.txt')
    targets = read_words(directory / 'level_words.txt')
    assert len(hints) > 2000
    assert set(hints) < set(targets)
    assert len(hints) == len(set(hints))
    manifest = json.loads((directory / 'build_manifest.json').read_text())
    assert len(hints) == manifest['hint_word_count']
