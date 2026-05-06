import sys, re
from pypdf import PdfReader

reader = PdfReader('report.pdf')
text_pages = [p.extract_text() for p in reader.pages]
text = '\n\n'.join(text_pages)

print(f'pages: {len(reader.pages)}')
print(f'extracted chars: {len(text)}')

# 1) literal '??' count (broken-reference fallback)
n_qq = text.count('??')
print(f'\n--- "??" count in PDF text: {n_qq} ---')
if n_qq:
    for m in re.finditer(r'.{0,40}\?\?.{0,40}', text):
        print('  ctx:', repr(m.group(0)))

# 2) fi-ligature smoke test: search for 'specific', 'unified',
#    'efficiency', 'classification', 'find', 'fix' (all with fi/fl ligatures).
#    With lmodern+T1 ligatures should extract as plain ASCII.
print('\n--- ligature words extraction smoke test ---')
for word in ['specific', 'unified', 'efficiency', 'classification',
             'finding', 'fixed', 'fine', 'fragmentation']:
    n = text.lower().count(word.lower())
    print(f'  {word!r:20s} -> {n} occurrences')

# 3) Show any non-ASCII chars in the extracted text
non_ascii = sorted({c for c in text if ord(c) > 127})
print(f'\n--- non-ASCII chars in extracted PDF text: {len(non_ascii)} ---')
for c in non_ascii[:30]:
    print(f'  U+{ord(c):04X} ({c!r}) count={text.count(c)}')

# 4) Sample around the word "specific" to confirm it's clean unicode
idx = text.lower().find('specific')
if idx >= 0:
    print(f'\n--- sample around first "specific": {text[max(0,idx-30):idx+30]!r} ---')
