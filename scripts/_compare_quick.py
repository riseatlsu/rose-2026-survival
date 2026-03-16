import re

def ex(text, pat, grp=1):
    m = re.search(pat, text)
    if not m: return '?'
    return m.group(grp).replace(',', '')

def diff(b, a):
    try:
        d = int(a) - int(b)
        return (f'+{d:,}' if d >= 0 else f'{d:,}'), (b != a)
    except:
        return '?', False

def row(label, b, a):
    d, changed = diff(b, a)
    flag = ' ◄' if changed else ''
    print(f'  {label:<28} before={int(b):>8,}  after={int(a):>8,}  {d:>8}{flag}')

b01 = open('audit_results/before/01_raw_json_author_types.txt').read()
a01 = open('audit_results/after/01_raw_json_author_types.txt').read()
b02 = open('audit_results/before/02_pipeline_transformation.txt').read()
a02 = open('audit_results/after/02_pipeline_transformation.txt').read()
b03 = open('audit_results/before/03_bot_detection_methods.txt').read()
a03 = open('audit_results/after/03_bot_detection_methods.txt').read()
b04 = open('audit_results/before/04_owner_type_coverage.txt').read()
a04 = open('audit_results/after/04_owner_type_coverage.txt').read()

print('=' * 65)
print('AUDIT 01 — Raw JSON commits (source files)')
print('=' * 65)
row('Total commits',   ex(b01,r'COMMITS \(([\d,]+) rec'), ex(a01,r'COMMITS \(([\d,]+) rec'))
row('User',            ex(b01,r'User\s+([\d,]+)\s+\(77'), ex(a01,r'User\s+([\d,]+)\s+\(7'))
row('Unknown',         ex(b01,r'Unknown\s+([\d,]+)\s+\(15'), ex(a01,r'Unknown\s+([\d,]+)\s+\(15'))
row('field absent',    ex(b01,r'absent\)\s+([\d,]+)\s+\(\d+\.\d+%\)\n  Bot'), ex(a01,r'absent\)\s+([\d,]+)\s+\(\d+\.\d+%\)\n  Bot'))
row('Bot',             ex(b01,r'Bot\s+([\d,]+)\s+\(0\.8'), ex(a01,r'Bot\s+([\d,]+)\s+\(0\.8'))
row('Organization',    ex(b01,r'Organization\s+(\d)\s+\(0\.00'), ex(a01,r'Organization\s+(\d)\s+\(0\.00'))

print()
print('=' * 65)
print('AUDIT 02 — Event CSVs (output of script 15)')
print('=' * 65)
for section, pat_label in [
    ('COMMITS',        r'COMMITS \(([\d,]+) rows\)'),
    ('ISSUES',         r'ISSUES \(([\d,]+) rows\)'),
    ('PULL_REQUESTS',  r'PULL_REQUESTS \(([\d,]+) rows\)'),
]:
    bt = ex(b02, pat_label); at = ex(a02, pat_label)
    print(f'  {section} — total: before={int(bt):,}  after={int(at):,}')
    for lbl, hp in [('Human', r'Human\s+([\d,]+)'), ('Bot', r'  Bot\s+([\d,]+)'), ('Organization', r'Organization\s+([\d,]+)')]:
        # find in section context
        def get_val(text, sec, p):
            idx = text.find(f'  {sec} (')
            chunk = text[idx:idx+300] if idx >= 0 else text
            return ex(chunk, p)
        bv = get_val(b02, section, hp)
        av = get_val(a02, section, hp)
        d, ch = diff(bv, av)
        flag = ' ◄' if ch else ''
        print(f'    {lbl:<15} before={int(bv):>8,}  after={int(av):>8,}  {d:>8}{flag}')

print()
print('=' * 65)
print('AUDIT 03 — Bot detection coverage')
print('=' * 65)
row('Total commits',        ex(b03,r'Total commits:\s+([\d,]+)'), ex(a03,r'Total commits:\s+([\d,]+)'))
row('With author_type API', ex(b03,r'With author_type from API:\s+([\d,]+)'), ex(a03,r'With author_type from API:\s+([\d,]+)'))
row('Without (field abs)',  ex(b03,r'Without \(rely on regex\):\s+([\d,]+)'), ex(a03,r'Without \(rely on regex\):\s+([\d,]+)'))

print()
print('=' * 65)
print('AUDIT 04 — Owner type')
print('=' * 65)
row('data=null repos',  ex(b04,r'With data=null:\s+(\d+)'), ex(a04,r'With data=null:\s+(\d+)'))
row('Organization',     ex(b04,r'Organization\s+(\d+)\s+\(8'), ex(a04,r'Organization\s+(\d+)\s+\(8'))
row('User',             ex(b04,r'  User\s+(\d+)\s+\(1'), ex(a04,r'  User\s+(\d+)\s+\(1'))
print(f'  {"Empty owner_type in dataset":<28}', 'BEFORE: present' if 'empty owner_type' not in b04 else 'BEFORE: none',
      '| AFTER:', 'All set ✓' if 'All repos have owner_type set' in a04 else 'still empty')
