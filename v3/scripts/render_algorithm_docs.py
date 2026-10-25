"""Build offline English HTML explanations with native MathML.

Requires Pandoc; pass --pandoc /path/to/pandoc when it is not on PATH.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pandoc', default=shutil.which('pandoc'))
    args = parser.parse_args()
    if not args.pandoc:
        parser.error('Pandoc is required; specify --pandoc /path/to/pandoc')
    docs = Path(__file__).resolve().parents[1] / 'docs'
    css = (docs / 'algorithm_document.css').read_text()
    with tempfile.TemporaryDirectory() as temporary:
        header = Path(temporary) / 'style.html'
        header.write_text('<style>\n' + css + '\n</style>')
        for suffix, language, title in (
            ('', 'en', 'PHASE I NEXUS algorithms and mathematical explanation'),
            ('_ZH', 'en', 'PHASE I NEXUS algorithms and mathematical explanation'),
        ):
            source = docs / ('V3_ALGORITHMS_AND_MATHEMATICAL_SPECIFICATION' + suffix + '.md')
            destination = source.with_suffix('.html')
            subprocess.run([args.pandoc, str(source), '--from=markdown+tex_math_dollars',
                '--to=html5', '--standalone', '--mathml', '--toc', '--toc-depth=2',
                '--metadata', 'pagetitle=' + title, '--metadata', 'lang=' + language,
                '--include-in-header', str(header), '--output', str(destination)], check=True)
            print(destination)


if __name__ == '__main__':
    main()
