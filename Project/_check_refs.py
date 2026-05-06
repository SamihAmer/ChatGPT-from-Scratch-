import re

with open('report.tex', encoding='utf-8') as f:
    text = f.read()

# 1) syntactic typos: \end{...> or \begin{...>
typos = re.findall(r'\\(?:begin|end)\{[^}]*>', text)
print('end/begin typos:', typos if typos else 'none')

# 2) all labels
labels = sorted(set(re.findall(r'\\label\{([^}]+)\}', text)))
print(f'\nlabels defined ({len(labels)} total):')
for l in labels:
    print(' ', l)

# 3) all refs
refs = sorted(set(re.findall(r'\\(?:ref|autoref|cref)\{([^}]+)\}', text)))
print(f'\nrefs used ({len(refs)} total):')
for r in refs:
    print(' ', r)

# 4) refs without matching label
broken = [r for r in refs if r not in labels]
print('\nBROKEN refs (no matching label):', broken if broken else 'none')

# 5) labels without any ref (informational)
unused = [l for l in labels if l not in refs]
print('unused labels:', unused if unused else 'none')
