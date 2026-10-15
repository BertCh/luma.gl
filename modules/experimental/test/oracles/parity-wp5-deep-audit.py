# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reproduce r9's real-GPU geomorphon error summaries from the exact definition.

This is historical evidence, NOT a new GPU execution. The three summary fields
must all match for each of 36 comparisons. Full after arrays are tested in the
GPU parity spec; summaries alone cannot establish pixelwise GPU equality.
"""
import hashlib
import json
from pathlib import Path
import sys

FIXTURES = Path(__file__).parent / 'fixtures'
EVALUATION = Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')


def generate():
    log = EVALUATION / 'gpu/r9.log'
    records = [json.loads(line[7:]) for line in log.read_text().splitlines() if line.startswith('PARITY ')]
    evidence = []
    for name in ['alps-dem', 'grand-canyon-dem', 'ridge']:
        matched = json.loads((FIXTURES / f'parity-wp5-deep-{name}-geomorphons.json').read_text())
        original = json.loads((FIXTURES / f'parity-wp5-round8-{name}-geomorphons.json').read_text())
        for matchedCase, originalCase in zip(matched['cases'], original['cases']):
            for metric in ['forms', 'ternary']:
                recorded = next(record for record in records if record['class'] == 'GPUGeomorphons'
                                and record['case'] == f'{name}/{matchedCase["name"]}' and record['metric'] == metric)
                differences = [abs(actual - expected) for actual, expected in zip(matchedCase[metric], originalCase[metric])]
                predicted = dict(mismatches=sum(value != 0 for value in differences), maxAbs=max(differences),
                                 maxRel=max(error / max(abs(value), sys.float_info.epsilon)
                                            for error, value in zip(differences, originalCase[metric])))
                assert all(recorded[key] == value for key, value in predicted.items()), (recorded, predicted)
                evidence.append(dict(recorded=recorded, predicted=predicted))
    payload = dict(conventions=__doc__, versions=dict(python=sys.version),
                   sourceLog=str(log), sourceLogSha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                   generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), comparisons=evidence)
    (FIXTURES / 'parity-wp5-deep-r9-audit.json').write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    print('All 36 r9 GPU summaries exactly reproduced: mismatches, maxAbs, maxRel.')


if __name__ == '__main__':
    generate()
