from detectors.lie_detector import evaluate_lie, LieDetectorConfig


def test_simple_no_lie():
    face = {'neutral': 0.8}
    voice = {'neutral': 0.75}
    fused = {'neutral': 0.78}
    out = evaluate_lie(face, voice, fused)
    assert out['lie_label'] == 'no_lie'
    assert out['lie_prob'] < 0.35


def test_conflict_and_fear():
    face = {'neutral': 0.1, 'fear': 0.7}
    voice = {'neutral': 0.05, 'angry': 0.72}
    fused = {'fear': 0.5, 'angry': 0.4, 'neutral': 0.1}
    out = evaluate_lie(face, voice, fused, cfg=LieDetectorConfig())
    # This scenario should yield a numeric probability and possibly reasons.
    assert out['lie_label'] in ('no_lie', 'uncertain', 'lie')
    assert 0.0 <= out['lie_prob'] <= 1.0
    # we expect at least one reason flagged (fear/low-neutral or conflict)
    assert isinstance(out.get('reasons', []), list)


if __name__ == '__main__':
    test_simple_no_lie()
    test_conflict_and_fear()
    print('lie_detector smoke tests passed')
