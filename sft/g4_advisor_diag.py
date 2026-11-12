# POST HOC ANALYSIS, EXPLORATORY. Written by the ML Research Scientist advisor; checked by the Thesis bot: run read-only on the Mac against the G4 attempt
# (20261111_203247_g4run.JoHtbc), exit 0; its class table, backend splits and pass counts (U1 88, T1 69) equal the independently computed figures in
# sft/g4_labels_calls.py and results log sections 14 and 15. It reads the benchmark's GOLD CALLS (possible_answer files) of the 200 test tasks and the frozen
# training rows (aggregate counts only, no task text printed). Gold calls are used only to describe; nothing here feeds data, a model or a retrain.
# Not changed from the advisor's original other than this header (the last line, Bp=, is unused). Paths are the Mac's.
# Advisor diagnostic of G4 (read-only). Aggregates only; prints no task text.
import json,os,glob,sqlite3,collections,re,statistics as st
H=os.path.expanduser
D=H("~/Documents/GitHub/Thesis/.venv/lib/python3.13/site-packages/bfcl_eval/data/")
E={}
for l in open(D+"BFCL_v4_multi_turn_base.json"):
    e=json.loads(l); E[e["id"]]=e
GT={}
for l in open(D+"possible_answer/BFCL_v4_multi_turn_base.json"):
    g=json.loads(l); GT[g["id"]]=g["ground_truth"]
name=lambda s: s.split("(")[0].strip()
G=sorted(glob.glob(H("~/Documents/GitHub/Thesis-wt-r6/sft/data/distill/r6g4/attempts/*g4run*")))[-1]
def load(r):
    c=sqlite3.connect("file:"+glob.glob(f"{G}/results_{r}/benchmark.db")[0]+"?mode=ro",uri=True)
    R={eid:json.loads(res) for eid,res in c.execute("select entry_id,result from responses")}
    P=set(json.load(open(f"{G}/RUN_{r}.json"))["passed_ids"]); return R,P
runs={r:load(r) for r in ("U1","T1")}
cls=collections.defaultdict(collections.Counter)
for i,e in E.items():
    for k in e["involved_classes"]:
        cls[k]["n"]+=1
        for r in runs: cls[k][r]+= i in runs[r][1]
print("CLASS n U1 T1")
for k,v in sorted(cls.items(), key=lambda x:-x[1]["n"]): print(k, v["n"], v["U1"], v["T1"])
norm=lambda s: re.sub(r"[\s\"']","",s)
gcalls=[len(t) for g in GT.values() for t in g]
print("BFCL gold: tasks",len(GT),"turns",len(gcalls),"mean calls/turn",round(st.mean(gcalls),2),"dist",sorted(collections.Counter(gcalls).items())[:9],"turns/task",dict(collections.Counter(len(g) for g in GT.values())))
for r,(R,P) in runs.items():
    cat=collections.Counter(); nrec=[0,0]; nprec=[0,0]; exact=[0,0]; steps=[]
    for i,g in GT.items():
        m=R.get(i,[])
        for t,gt in enumerate(g):
            mt=m[t] if t<len(m) else []
            flat=[c for s in mt for c in (s if isinstance(s,list) else [s])]
            steps.append(len(mt)); mc=len(flat); gc=len(gt)
            cat["gold0" if gc==0 else "empty" if mc==0 else "fewer" if mc<gc else "equal" if mc==gc else "more"]+=1
            gn=collections.Counter(name(x) for x in gt); mn=collections.Counter(name(x) for x in flat)
            nrec[0]+=sum((gn&mn).values()); nrec[1]+=sum(gn.values()); nprec[0]+=sum((gn&mn).values()); nprec[1]+=sum(mn.values())
            ms=set(norm(x) for x in flat); exact[0]+=sum(norm(x) in ms for x in gt); exact[1]+=len(gt)
    print(r,"pass",len(P),"turn cats",dict(cat),"name recall",round(nrec[0]/nrec[1],3),"name precision",round(nprec[0]/nprec[1],3),"exact gold-call found",round(exact[0]/exact[1],3),"model calls",nprec[1],"gold calls",nrec[1],"mean steps/turn",round(st.mean(steps),2))
L=[json.loads(l) for l in open(H("~/Documents/GitHub/Thesis-wt-r6/sft/data/distill/r6g3/final/A/train.jsonl"))]
print("meta keys",list(L[0]["meta"].keys()))
print("actions",dict(collections.Counter(x["meta"]["action"] for x in L)),"n_calls",sorted(collections.Counter(x["meta"]["n_calls"] for x in L).items()))
tasks={}
for x in L: tasks.setdefault(x["meta"]["task_id"],x["meta"])
tc=collections.Counter(k for m in tasks.values() for k in m["classes"]); print("train tasks",len(tasks),"class counts",dict(tc))
bc=collections.Counter(k for e in E.values() for k in e["involved_classes"]); print("BFCL class counts",dict(bc))
fn=collections.Counter()
for x in L:
    try:
        for c in json.loads(x["messages"][-1]["content"]): fn[c["function_name"]]+=1
    except Exception: pass
gfn=collections.Counter(name(c) for g in GT.values() for t in g for c in t)
tot=sum(gfn.values()); print("train distinct fns",len(fn),"train target calls",sum(fn.values()),"BFCL gold distinct fns",len(gfn),"gold calls",tot)
for k in (1,5,20): print("share of BFCL gold calls whose function has >=%d training targets:"%k,round(sum(v for f,v in gfn.items() if fn[f]>=k)/tot,3),"distinct fns below:",sum(1 for f in gfn if fn[f]<k))
print("top BFCL gold fns (gold, train):",[(f,v,fn[f]) for f,v in gfn.most_common(16)])
print("top train fns:",fn.most_common(12))
ep=collections.Counter(x["meta"]["episode_id"] for x in L); print("rows/episode mean",round(st.mean(ep.values()),2),"episodes",len(ep),"episodes/task",round(len(ep)/len(tasks),2))
uw=[len(q[0]["content"].split()) for e in E.values() for q in e["question"] if q]; print("BFCL user turn words mean",round(st.mean(uw),1),"median",st.median(uw))
print("---- follow-up")
U=runs["U1"][1]; T=runs["T1"][1]
veh={i for i,e in E.items() if "VehicleControlAPI" in e["involved_classes"]}
for nm,S in (("vehicle tasks",veh),("non-vehicle tasks",set(E)-veh)):
    print(nm,len(S),"U1",len(U&S),"T1",len(T&S),"untuned-only",len((U-T)&S),"trained-only",len((T-U)&S))
pc=collections.Counter(); 
for i in U-T:
    for k in E[i]["involved_classes"]: pc["U-only "+k]+=1
for i in T-U:
    for k in E[i]["involved_classes"]: pc["T-only "+k]+=1
print(dict(pc))
# coverage score per task: min over gold fns of training count
def cov(i): return min((fn[name(c)] for t in GT[i] for c in t), default=0)
def lowshare(i):
    cs=[fn[name(c)] for t in GT[i] for c in t]; return sum(1 for x in cs if x<5)/max(1,len(cs))
for lo,hi,lab in ((0,0.0001,"all gold fns seen >=5 times in training"),(0.0001,0.5,"some rare/unseen (<50% of calls)"),(0.5,1.01,"mostly rare/unseen (>=50% of calls)")):
    S={i for i in E if lo<=lowshare(i)<hi}
    print(lab,":",len(S),"tasks; U1",len(U&S),"T1",len(T&S),"untuned-only",len((U-T)&S),"trained-only",len((T-U)&S))
# vehicle function accuracy
for r,(R,P) in runs.items():
    c=collections.Counter()
    for i in veh:
        m=R.get(i,[])
        for t,gt in enumerate(GT[i]):
            mt=m[t] if t<len(m) else []
            flat=[name(x) for s in mt for x in (s if isinstance(s,list) else [s])]
            for f in ("startEngine","pressBrakePedal","lockDoors"):
                if any(name(x)==f for x in gt): c[f+" gold turns"]+=1; c[f+" called"]+= f in flat
    print(r,dict(c))
Bp=H("~/Documents/GitHub/Thesis-wt-r6/sft/data/distill/r6g3/final/B/train.jsonl")
