# POST HOC ANALYSIS, EXPLORATORY. Written by the ML Research Scientist advisor; checked by the Thesis bot: run read-only on the Mac against the G4 attempt
# (20261111_203247_g4run.JoHtbc), exit 0; its class table, backend splits and pass counts (U1 88, T1 69) equal the independently computed figures in
# the function-count lines of results log sections 14 and 15 (checked: arm A function counts for startEngine, pressBrakePedal, lockDoors equal 2, 2, 5 here). It reads the benchmark's GOLD CALLS (possible_answer files) of the 200 test tasks and the frozen
# training rows (aggregate counts only, no task text printed). Gold calls are used only to describe; nothing here feeds data, a model or a retrain.
# Not changed from the advisor's original other than this header (the last line, Bp=, is unused). Paths are the Mac's.
# coverage of the BFCL tool surface by training targets (aggregates only)
import json,os,glob,collections
H=os.path.expanduser
D=H("~/Documents/GitHub/Thesis/.venv/lib/python3.13/site-packages/bfcl_eval/data/")
name=lambda s: s.split("(")[0].strip()
GT=[json.loads(l)["ground_truth"] for l in open(D+"possible_answer/BFCL_v4_multi_turn_base.json")]
gfn=collections.Counter(name(c) for g in GT for t in g for c in t); tot=sum(gfn.values())
docs={}
for f in glob.glob(D+"multi_turn_func_doc/*.json"):
    k=os.path.basename(f)[:-5]
    try: docs[k]=[json.loads(l)["name"].split(".")[-1] for l in open(f) if l.strip()]
    except Exception as e: docs[k]=[]
base=("gorilla_file_system","vehicle_control","trading_bot","travel_booking","message_api","posting_api","twitter_api","ticket_api","math_api")
allf={f:k for k in docs for f in docs[k] if k in base}
print("doc files",{k:len(v) for k,v in docs.items()})
def fnc(path):
    fn=collections.Counter(); n=0; tasks=set(); cl=collections.Counter()
    for l in open(path):
        x=json.loads(l); n+=1
        if x["meta"]["task_id"] not in tasks:
            tasks.add(x["meta"]["task_id"]); cl.update(x["meta"]["classes"])
        try:
            j=json.loads(x["messages"][-1]["content"].split("\n\n")[-1]) if x["messages"][-1]["content"].lstrip().startswith("[") else None
            for c in (j or []): fn[c["function_name"]]+=1
        except Exception: pass
    return fn,n,len(tasks),cl
for lab,p in (("armA batch1","~/Documents/GitHub/Thesis-wt-r6/sft/data/distill/r6g3/final/A/train.jsonl"),("combined A (retrain)","~/Documents/GitHub/Thesis-wt-r6/sft/data/distill/r6g3c/final/A/train.jsonl")):
    fn,n,nt,cl=fnc(H(p))
    print("==",lab,"rows",n,"tasks",nt,"classes",dict(cl))
    for k in (1,5,20): print("  BFCL gold calls on functions with >=%d training targets: %.3f"%(k,sum(v for f,v in gfn.items() if fn[f]>=k)/tot))
    for k in sorted(set(allf.values())):
        fs=[f for f in allf if allf[f]==k]; print("  ",k,"functions in docs",len(fs),"with 0 targets",sum(fn[f]==0 for f in fs),"with <5",sum(fn[f]<5 for f in fs))
    print("  vehicle chain targets:",{f:fn[f] for f in ("startEngine","pressBrakePedal","lockDoors","releaseBrakePedal","setCruiseControl","activateParkingBrake")},"| travel:",{f:fn[f] for f in ("book_flight","get_flight_cost","get_zipcode_based_on_city","retrieve_invoice","cancel_booking")},"| msg:",{f:fn[f] for f in ("send_message","view_messages_sent","get_user_id","message_login")},"| fs:",{f:fn[f] for f in ("touch","mkdir","echo","mv","grep","sort","diff")})
