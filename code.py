"""
WEEK 6: Supply Chain Risk Analysis and Strategic Reporting
Run: pip install numpy pandas matplotlib  then  python week6_code.py
PROCESS
1. Define network: 3 suppliers, 2 warehouses, 5 regions.
2. List 10 risks; score Probability (1-5) and Impact (1-5); Score = P x I.
   Levels: Critical 15+, High 10-14, Medium 6-9, Low below 6.
3. Quantify: annual probability, average loss (INR lakh), Expected Annual Loss = probability x loss.
4. Monte Carlo: simulate 20,000 years, get mean, P90, P95 of total loss.
5. Mitigation: assumed risk reduction and yearly cost; net benefit = saving - cost. Re-run simulation after mitigation.
OUTPUT: output/week6_risk_register_dataset.csv, output/results_week6.json, output/img/w6_*.png
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.makedirs('output/img',exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
NAVY='#1F3A5F'; TEAL='#2A9D8F'; ORG='#E76F51'; GOLD='#E9C46A'; GREY='#8D99AE'
R={}
# ================= WEEK 6 =================
rng=np.random.default_rng(11)
risks=[ # id,name,category,prob(1-5),impact(1-5),prob_annual,loss_mean(lakh INR),loss_sd
('R1','Port congestion / transportation delay','Transportation',4,4,0.60,18,6),
('R2','Fuel price spike','Cost',4,3,0.50,12,4),
('R3','Inventory stock-out at warehouse','Inventory',3,5,0.35,30,10),
('R4','Supplier failure / late supply','Supplier',3,4,0.30,25,8),
('R5','Warehouse disruption (fire, flood, IT outage)','Facility',2,5,0.10,60,20),
('R6','Labour strike / driver shortage','Labour',3,3,0.30,10,3),
('R7','Cyber attack on tracking / WMS system','Technology',2,4,0.12,35,12),
('R8','Cargo damage and theft in transit','Transportation',3,2,0.45,6,2),
('R9','Regulatory / customs change','Compliance',2,3,0.20,8,3),
('R10','Demand surge beyond forecast','Demand',3,3,0.40,14,5)]
rk=pd.DataFrame(risks,columns=['id','risk','category','prob_score','impact_score','p_annual','loss_mean','loss_sd'])
rk['score']=rk.prob_score*rk.impact_score; rk['eal']=rk.p_annual*rk.loss_mean
rk['level']=np.where(rk.score>=15,'Critical',np.where(rk.score>=10,'High',np.where(rk.score>=6,'Medium','Low')))
sims=20000; tot=np.zeros(sims)
for _,r in rk.iterrows():
    occ=rng.random(sims)<r.p_annual; tot+=occ*np.maximum(0,rng.normal(r.loss_mean,r.loss_sd,sims))
mit={'R1':(0.50,3.0),'R2':(0.40,2.0),'R3':(0.55,3.5),'R4':(0.50,3.0),'R5':(0.50,2.5),'R6':(0.40,1.0),'R7':(0.60,2.0),'R8':(0.40,1.0),'R9':(0.30,1.0),'R10':(0.40,2.0)}
rk['reduction']=rk.id.map(lambda i:mit[i][0]); rk['mit_cost']=rk.id.map(lambda i:mit[i][1]); rk['saving']=rk.eal*rk.reduction; rk['net']=rk.saving-rk.mit_cost
tot2=np.zeros(sims)
for _,r in rk.iterrows():
    occ=rng.random(sims)<r.p_annual*(1-r.reduction); tot2+=occ*np.maximum(0,rng.normal(r.loss_mean,r.loss_sd,sims))
R['w6']={'risks':rk.round(2).to_dict('records'),'eal_total':float(rk.eal.sum()),'mean':float(tot.mean()),'p50':float(np.percentile(tot,50)),'p90':float(np.percentile(tot,90)),'p95':float(np.percentile(tot,95)),
 'mean_m':float(tot2.mean()),'p90_m':float(np.percentile(tot2,90)),'p95_m':float(np.percentile(tot2,95)),'mit_cost':float(rk.mit_cost.sum()),'saving':float(rk.saving.sum()),'net':float(rk.net.sum()),'roi':float(rk.net.sum()/rk.mit_cost.sum()*100)}
rk.to_csv('output/week6_risk_register_dataset.csv',index=False)
# charts
fig,ax=plt.subplots(figsize=(6,4.6)); M=np.zeros((5,5))
ax.set_xlim(.5,5.5); ax.set_ylim(.5,5.5)
for p in range(1,6):
    for i in range(1,6):
        s=p*i; c='#F4B6A9' if s>=15 else '#F9D7A0' if s>=10 else '#FBEFB0' if s>=6 else '#CFE8D5'
        ax.add_patch(plt.Rectangle((i-.5,p-.5),1,1,fc=c,ec='white'))
for _,r in rk.iterrows():
    same=rk[(rk.prob_score==r.prob_score)&(rk.impact_score==r.impact_score)]; k=list(same.id).index(r.id)
    ax.text(r.impact_score-.3+.3*k,r.prob_score,r.id,fontsize=8.5,weight='bold',ha='center',va='center')
ax.set_xlabel('Impact (1 low - 5 severe)'); ax.set_ylabel('Probability (1 rare - 5 almost certain)'); ax.set_title('Risk Heat Map'); ax.set_xticks(range(1,6)); ax.set_yticks(range(1,6)); plt.tight_layout(); plt.savefig('output/img/w6_heat.png'); plt.close()
s=rk.sort_values('eal'); fig,ax=plt.subplots(figsize=(7,3.8)); ax.barh(s.id+' '+s.risk.str.slice(0,32),s.eal,color=NAVY); ax.set_xlabel('Expected annual loss (INR lakh)'); ax.set_title('Expected Annual Loss by Risk'); plt.tight_layout(); plt.savefig('output/img/w6_eal.png'); plt.close()
fig,ax=plt.subplots(figsize=(7,3.4)); ax.hist(tot,bins=60,alpha=.7,color=ORG,label='Before mitigation'); ax.hist(tot2,bins=60,alpha=.7,color=TEAL,label='After mitigation'); ax.axvline(np.percentile(tot,95),color=ORG,ls='--'); ax.axvline(np.percentile(tot2,95),color=TEAL,ls='--'); ax.set_xlabel('Total annual loss (INR lakh)'); ax.set_title('Monte Carlo Simulation (20,000 runs): Annual Risk Loss'); ax.legend(); plt.tight_layout(); plt.savefig('output/img/w6_mc.png'); plt.close()
# network diagram
fig,ax=plt.subplots(figsize=(8,3.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,50)
def nb(x,y,t,c): ax.add_patch(FancyBboxPatch((x,y),16,8,boxstyle='round,pad=0.3',fc=c,ec='k')); ax.text(x+8,y+4,t,ha='center',va='center',fontsize=8)
S=[(2,38,'Supplier A\n(Port Mumbai)'),(2,21,'Supplier B\n(Port Chennai)'),(2,4,'Supplier C\n(Domestic)')]
W=[(40,30,'Warehouse 1\n(Bengaluru)'),(40,10,'Warehouse 2\n(Hyderabad)')]
C=[(80,40,'Region North'),(80,30,'Region South'),(80,20,'Region East'),(80,10,'Region West'),(80,0,'Region Central')]
for x,y_,t in S: nb(x,y_,t,'#DCE6F7')
for x,y_,t in W: nb(x,y_,t,'#FFF2CC')
for x,y_,t in C: nb(x,y_,t,'#D9EFE3')
for _,y1,_t in S:
    for _,y2,_t2 in W: ax.annotate('',xy=(39.5,y2+4),xytext=(18.8,y1+4),arrowprops=dict(arrowstyle='->',color=GREY,lw=.8))
for _,y1,_t in W:
    for _,y2,_t2 in C: ax.annotate('',xy=(79.5,y2+4),xytext=(56.8,y1+4),arrowprops=dict(arrowstyle='->',color=GREY,lw=.8))
ax.text(10,48,'Inbound (Suppliers)',ha='center',weight='bold',fontsize=8); ax.text(48,44,'Storage (Warehouses)',ha='center',weight='bold',fontsize=8); ax.text(88,48,'Outbound (Customers)',ha='center',weight='bold',fontsize=8)
plt.tight_layout(); plt.savefig('output/img/w6_network.png'); plt.close()

json.dump(R,open('output/results_week6.json','w'),indent=1,default=float)
print('Done. Check the output folder.')