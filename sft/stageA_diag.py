import json,re,sys
sys.path.insert(0,'.')
NOTE="Note: at least one call above returned an error."
F='sft/data/distill/f/'
arms=['B1','B2','H','AB']
V={a:{x['task_id']:x for x in map(json.loads,open(f'{F}eval_{a}/verdicts.jsonl'))} for a in arms}
tasks=sorted(V['B1'])
def passes(a): return {t for t,x in V[a].items() if x['valid']}
P={a:passes(a) for a in arms}
print("pass",{a:len(P[a]) for a in arms})
def paired(a,b): return (len(P[a]-P[b]),len(P[b]-P[a]))  # gains of a over b, losses
for a,b in [('H','B1'),('H','B2'),('AB','B1'),('AB','B2'),('B2','B1')]:
    g,l=paired(a,b); print(f"{a} vs {b}: gains {g} losses {l}")
print("B1,B2 identical pass sets:",P['B1']==P['B2'])
# cost diagnostics
for a in arms:
    E=[json.loads(l) for l in open(f'{F}eval_{a}/episodes.jsonl')]
    print(a,"api_calls",sum(V[a][t]['api_calls'] for t in V[a]),"prompt_tok",sum(e['prompt_tokens'] for e in E),"compl_tok",sum(e['completion_tokens'] for e in E),"wall_s_sum",round(sum(e['wall_s'] for e in E)),)
# post-error step analysis
def is_empty(txt):
    t=re.sub(r'\s+','',txt or '')
    return t in ('[]','')
from benchmark.iffc_state import is_error_result
def exec_msg(m):
    """Execution results of the step the model is now replying to: they sit inside the LAST user message (after the OODA card prefix)."""
    if not m or m[-1].get('role')!='user': return None
    c=str(m[-1].get('content') or '')
    k=c.find('Function execution results:')
    return c[k:] if k>=0 else None
def msg_has_error(c):
    for line in c.splitlines():
        if line.startswith('- ') and ' -> ' in line:
            if is_error_result(line.split(' -> ',1)[1]): return True
    return False
def post_error(a):
    C=[json.loads(l) for l in open(f'{F}eval_{a}/calls.jsonl')]
    n=emp=0; eps=set(); marker=0; agree=0; dis=0
    for c in C:
        m=c['messages']
        # the call directly follows an executed step iff the latest exec-results message is within the last 3 messages
        em=exec_msg(m)
        det = bool(em) and msg_has_error(em)
        has_note = bool(em) and NOTE in em
        if a=='H':
            if det==has_note: agree+=1
            else: dis+=1
        if det:
            n+=1; eps.add(c['episode_id']); emp+=is_empty(c['response_text'])
        marker+=has_note
    if a=='H': print("  H detector vs note agreement:",agree,"disagree",dis)
    return n,emp,eps,marker
base_err={}
for a in arms:
    n,emp,eps,mk=post_error(a)
    print(a,"post-error steps",n,"empty",emp,"new-call",n-emp,"episodes with error",len(eps),"note-marker calls",mk)
    base_err[a]=eps
# error / no-error subsets defined on baseline B1 (episode_id -> task)
tid=lambda e:e.split('__')[0]
errB1={tid(e) for e in base_err['B1']}
for a in arms:
    e=[t for t in tasks if t in errB1]; ne=[t for t in tasks if t not in errB1]
    print(a,"B1-error subset (n=%d) pass"%len(e),sum(t in P[a] for t in e),"| no-error (n=%d) pass"%len(ne),sum(t in P[a] for t in ne))
# declared subset (Round 4 plan): episodes error-free in BOTH base runs vs error in either base run (A0.2, Round 5)
errB2={tid(e) for e in base_err['B2']}
err_any=errB1|errB2
print("declared subsets: error in B1 or B2 (n=%d), error-free in both (n=%d)"%(len([t for t in tasks if t in err_any]),len([t for t in tasks if t not in err_any])))
for a in arms:
    e=[t for t in tasks if t in err_any]; ne=[t for t in tasks if t not in err_any]
    print(a,"error-in-either-base subset (n=%d) pass"%len(e),sum(t in P[a] for t in e),"| error-free-in-both (n=%d) pass"%len(ne),sum(t in P[a] for t in ne))
