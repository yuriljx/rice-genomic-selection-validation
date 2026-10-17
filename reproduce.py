import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from inputs import read_wei, read_irri

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()

def correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = a - np.mean(a)
    y = b - np.mean(b)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0 else float(np.clip(np.dot(x, y) / denominator, -1., 1.))

def ranks(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result

def metrics(ids, pred, truth, percent, irri=False, include_spearman=True):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if pred.shape != y.shape or len(ids) != len(pred) or len(ids) != len(set(ids)) or not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError('Invalid scoring arrays or IDs')
    order = sorted(range(len(ids)), key=lambda i: (-float(pred[i]), digest(('selection-v1|2026100503|' + ids[i]).encode()), ids[i]))
    q = math.ceil(percent / 100 * len(ids))
    top = order[:q]
    mean = (lambda values: math.fsum(float(x) / len(values) for x in values)) if irri else np.mean
    base = float(mean(y))
    selected = float(mean(y[top]))
    return {'N_S':len(ids), 'q':q, 'S_mean':base, 'selected_mean':selected, 'U':selected-base, 'r':correlation(pred,y), 'Spearman':correlation(ranks(pred),ranks(y)) if include_spearman else None, 'selected_ID_order_sha256':digest(canonical([ids[i] for i in top]))}

def rowkey(row):
    names = ('dataset','task','view','arm','policy','season','configuration','lambda','q_percent')
    return '|'.join(str(row.get(k,'')) if row.get(k) is not None else '' for k in names)

def description(values):
    values = list(values)
    if not values or any(v is None or not math.isfinite(float(v)) for v in values):
        return {'n_defined':sum(v is not None for v in values), 'n_total':len(values)}
    return {'n_defined':len(values), 'n_total':len(values), 'mean':float(np.mean(values)), 'median':float(np.median(values)), 'sample_SD':float(np.std(values,ddof=1)) if len(values)>1 else 0., 'min':min(values), 'max':max(values), 'positive':sum(v>0 for v in values), 'zero_exact':sum(v==0 for v in values), 'negative':sum(v<0 for v in values)}

def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows, resampling):
    look = {(r['dataset'],r['task'],r['view'],r['arm'],r['policy'],r['season'],r['configuration'],r['lambda'],r['q_percent']):r for r in rows}
    family_groups=collections.defaultdict(list)
    other_groups=collections.defaultdict(list)
    contrasts={'NAM_four_arm':[('delta_L','pool100','ridge0'),('delta_G','rkhs100','rkhs0'),('K0','rkhs0','ridge0'),('K100','rkhs100','pool100'),('D','pool100','rkhs0')], 'NAM_original':[('original_primary','weighted100','rkhs0'),('pooling_check','pool100','ridge0')], 'NAM_fixed_grid':[('pooling','pool100','ridge0')], 'MAGIC_original':[('original_primary','weighted100','rkhs0'),('pooling_check','pool100','ridge0'),('calibrated_Gaussian','pool_rkhs100','rkhs0'),('calibrated_kernel','pool_rkhs100','pool100'),('source_kernel','rkhs0','ridge0'),('weighting','weighted100','pool100'),('weighted_vs_calibrated_Gaussian','weighted100','pool_rkhs100'),('weighted_vs_ridge','weighted100','ridge0'),('weighted_vs_target_only','weighted100','source_tuned_targetonly100'),('C25_vs_Gaussian','weighted25','rkhs0')], 'IRRI_original':[('original_primary','pool25','RKHS0'),('ridge_comparator','pool25','ridge0')], 'IRRI_partitions':[('pool_vs_Gaussian','pool25','RKHS0'),('pool_vs_ridge','pool25','ridge0'),('source_kernel','RKHS0','ridge0'),('weighting','weighted25','pool25'),('calibrated_kernel','pooledRKHS25','pool25'),('Gaussian_calibration','pooledRKHS25','RKHS0')]}
    for row in sorted(rows,key=lambda x:(x['dataset'],x['view'],x['task'],x['arm'],x['q_percent'])):
        ds=row['dataset']
        for contrast,a,b in contrasts[ds]:
            if row['arm']!=a:
                continue
            key=(ds,row['task'],row['view'],b,row['policy'],row['season'],row['configuration'],row['lambda'],row['q_percent'])
            if ds in ['NAM_original','MAGIC_original','IRRI_original']:
                candidates=[r for r in rows if (r['dataset'],r['task'] if ds!='MAGIC_original' else r['draw'],r['view'],r['arm'],r['q_percent'])==(ds,row['task'] if ds!='MAGIC_original' else row['draw'],row['view'],b,row['q_percent'])]
                if len(candidates)!=1:
                    continue
                baseline=candidates[0]
            else:
                baseline=look.get(key)
                if baseline is None:
                    candidates=[r for r in rows if all(r[k]==row[k] for k in ['dataset','task','view','policy','season','configuration','q_percent']) and r['arm']==b]
                    if len(candidates)!=1:
                        raise ValueError('Missing or ambiguous paired baseline')
                    baseline=candidates[0]
            for metric in (['U','r'] if ds in ['NAM_fixed_grid','MAGIC_original','IRRI_original'] else ['U','r','Spearman']):
                x=row.get(metric);y=baseline.get(metric)
                value=None if x is None or y is None else x-y
                k=(ds,row['view'],row['policy'],row['season'],row['lambda'] if ds=='NAM_fixed_grid' else None,row['q_percent'],contrast,metric)
                if row['family']:
                    family_groups[k+(row['family'],)].append(value)
                else:
                    other_groups[k].append(value)
    for key,values in list(family_groups.items()):
        if key[0]=='NAM_four_arm' and key[6]=='delta_G':
            other=key[:6]+('delta_L',)+key[7:]
            family_groups[key[:6]+('I',)+key[7:]]=[None if a is None or b is None else a-b for a,b in zip(values,family_groups[other])]
    families=[];groups=collections.defaultdict(list)
    for key,values in family_groups.items():
        ds,view,policy,season,lam,q,contrast,metric,family=key
        if len(values)!=5:
            raise ValueError('Expected five paired draws in every family')
        v=None if any(x is None for x in values) else float(np.mean(values))
        families.append({'dataset':ds,'view':view,'policy':policy,'season':season,'lambda':lam,'q_percent':q,'contrast':contrast,'metric':metric,'family':family,'five_draw_mean':v})
        groups[key[:-1]].append((family,v))
    for key,values in list(other_groups.items()):
        if key[0]=='MAGIC_original' and key[6]=='calibrated_kernel':
            source=key[:6]+('source_kernel',)+key[7:]
            other_groups[key[:6]+('I',)+key[7:]]=[None if a is None or b is None else a-b for a,b in zip(values,other_groups[source])]
    summaries=[]
    for key,fv in groups.items():
        ds,view,policy,season,lam,q,contrast,metric=key
        fv=sorted(fv,key=lambda x:int(x[0][1:]))
        if [f for f,v in fv]!=['N'+str(i) for i in range(1,16)]:
            raise ValueError('Expected the full fifteen-family lattice')
        values=[v for f,v in fv]
        row=dict(zip(['dataset','view','policy','season','lambda','q_percent','contrast','metric'],key))
        row.update(description(values))
        if row['n_defined']==15 and q==10 and (ds=='NAM_four_arm' and metric in ['U','r'] or ds=='NAM_original' and metric=='U'):
            a=np.asarray(values);means=np.mean(a[resampling],axis=1)
            row.update(bootstrap_low=float(np.quantile(means,.025,method='linear')),bootstrap_high=float(np.quantile(means,.975,method='linear')),t14_low=row['mean']-2.144786687916927*float(np.std(a,ddof=1))/math.sqrt(15),t14_high=row['mean']+2.144786687916927*float(np.std(a,ddof=1))/math.sqrt(15))
        if row['n_defined']==15 and ds=='NAM_four_arm' and q==10 and metric=='r' and view in ['SH','HZ'] and contrast in ['delta_L','delta_G','D']:
            a=np.asarray(values);means=np.mean(a[resampling],axis=1);half=3.068779245476067*float(np.std(a,ddof=1))/math.sqrt(15)
            row.update(selected_six_bootstrap_low=float(np.quantile(means,.05/12,method='linear')),selected_six_bootstrap_high=float(np.quantile(means,1-.05/12,method='linear')),selected_six_t14_low=row['mean']-half,selected_six_t14_high=row['mean']+half)
        summaries.append(row)
    for key,values in other_groups.items():
        row=dict(zip(['dataset','view','policy','season','lambda','q_percent','contrast','metric'],key));row.update(description(values));summaries.append(row)
    return families,summaries

def execute(args):
    root=Path(__file__).resolve().parent
    resources=root/'resources'
    spec=json.loads((root/'study.json').read_text())
    for name,ref in spec['resources'].items():
        p=root/name
        if p.stat().st_size!=ref['bytes'] or digest(p.read_bytes())!=ref['sha256']:
            raise ValueError('Resource identity differs: '+name)
    study=json.loads(gzip.decompress((resources/'study.json.gz').read_bytes()))
    if collections.Counter(c['dataset'] for c in study['cells'])!=study['cell_counts']:
        raise ValueError('Task lattice differs')
    required_wei=set();required_irri=set()
    sets=study['ID_sets']
    for c in study['cells']:
        ids=sets[c['score_set']]
        if set(ids)&set(sets[c['calibration_set']]):
            raise ValueError('Scoring and calibration roles overlap')
        dest=required_irri if c['dataset'].startswith('IRRI') else required_wei
        dest.update((i,c['season'] if c['dataset'].startswith('IRRI') else c['site']) for i in ids)
    out=Path(args.out).resolve()
    if out.exists() or out==root or root.is_relative_to(out):
        raise ValueError('Use a new output directory')
    truth=read_wei(Path(args.wei_xlsx),resources,required_wei)
    irri=read_irri(Path(args.irri_plotzip),required_irri)
    out.mkdir(parents=True,exist_ok=False)
    rows=[];cache={}
    with np.load(resources/'predictions.npz',allow_pickle=False) as predictions:
        if len(predictions.files)!=study['unique_prediction_vectors']:
            raise ValueError('Prediction array inventory differs')
        for cell in study['cells']:
            ids=sets[cell['score_set']]
            pair=(cell['prediction_set'],cell['score_set'])
            if pair not in cache:
                positions={i:k for k,i in enumerate(sets[pair[0]])}
                cache[pair]=[positions[i] for i in ids]
            pred=predictions[cell['prediction_key']][cache[pair]]
            is_irri=cell['dataset'].startswith('IRRI')
            y=[(irri if is_irri else truth)[(i,cell['season'] if is_irri else cell['site'])] for i in ids]
            fractions=[5,10,20] if cell['dataset'] in ['NAM_four_arm','IRRI_partitions'] or cell['dataset']=='NAM_original' and cell['arm'] in ['weighted100','rkhs0'] else [10]
            for q in fractions:
                m=metrics(ids,pred,y,q,is_irri,cell['dataset'] not in ['NAM_fixed_grid','MAGIC_original','IRRI_original'])
                metadata={k:cell[k] for k in ['dataset','task','view','site','family','draw','arm','policy','season','configuration','lambda']}
                rows.append({**metadata,'q_percent':q,**m})
    references=list(csv.DictReader(io.StringIO(gzip.decompress((resources/'reference.csv.gz').read_bytes()).decode())))
    actual={rowkey(r):r for r in rows}
    if len(actual)!=len(rows):
        raise ValueError('Duplicate output identity')
    worst={};checks=0
    for ref in references:
        row=actual.get(ref['key'])
        if row is None:
            raise ValueError('Reference cell is missing: '+ref['key'])
        for name in ['N_S','q','U','r','Spearman']:
            v=ref.get(name,'')
            if v=='':
                continue
            x=row[name]
            if x is None:
                raise ValueError('Defined reference became undefined')
            error=abs(float(x)-float(v))
            tolerance=0 if name in ['N_S','q'] else 1e-8 if name=='U' else 1e-10
            worst[name]=max(worst.get(name,0),error)
            if error>tolerance:
                raise ValueError('Reference mismatch: '+ref['key']+' '+name)
            checks+=1
        selected=ref.get('selected_ID_order_sha256','')
        if selected:
            if row['selected_ID_order_sha256']!=selected:raise ValueError('Original selected ID order differs: '+ref['key'])
            checks+=1
    resampling=np.load(resources/'resampling.npy',allow_pickle=False)
    if resampling.shape!=(10000,15) or np.any(resampling<0) or np.any(resampling>14):
        raise ValueError('Original resampling matrix differs')
    families,summaries=summarize(rows,resampling)
    magic_reference=list(csv.DictReader((resources/'MAGIC_contrast_reference.csv').open()))
    for ref in magic_reference:
        label=spec['MAGIC_contrast_map'][ref['contrast']]
        matches=[v for v in summaries if (v['dataset'],v['view'],v['contrast'],v['q_percent'],v['metric'])==('MAGIC_original',ref['site'],label,10,'U')]
        if len(matches)!=1:raise ValueError('Missing original MAGIC contrast')
        for field,original in [('mean','mean_delta_U'),('min','min_delta_U'),('max','max_delta_U')]:
            error=abs(matches[0][field]-float(ref[original]))
            if error>1e-10:raise ValueError('Original MAGIC contrast differs')
            worst['MAGIC_contrast']=max(worst.get('MAGIC_contrast',0),error)
            checks+=1

    ag=collections.defaultdict(list)
    for row in rows:
        key=(row['dataset'],row['view'],row['policy'],row['season'],row['lambda'] if row['dataset']=='NAM_fixed_grid' else None,row['q_percent'],row['arm'])
        if row['family']:
            key=key+(row['family'],)
        ag[key].append(row)
    family_arms=[]
    arm_groups=collections.defaultdict(list)
    for key,rr in ag.items():
        values={m:None if any(r[m] is None for r in rr) else float(np.mean([r[m] for r in rr])) for m in ['U','r','Spearman']}
        base=dict(zip(['dataset','view','policy','season','lambda','q_percent','arm'],key[:7]))
        if len(key)==8:
            if len(rr)!=5:
                raise ValueError('Arm family is missing one of five draws')
            family_arms.append({**base,'family':key[7],**values})
        if len(key)==8:
            arm_groups[key[:7]].append(values)
        else:
            arm_groups[key[:7]].extend({m:row[m] for m in ['U','r','Spearman']} for row in rr)
    arm_summaries=[]
    for key,rr in arm_groups.items():
        base=dict(zip(['dataset','view','policy','season','lambda','q_percent','arm'],key))
        for metric in ['U','r','Spearman']:
            if metric=='Spearman' and all(row[metric] is None for row in rr):continue
            arm_summaries.append({**base,'metric':metric,**description([r[metric] for r in rr])})
    dispersion=[]
    for site in ['SH','HZ']:
        for contrast,baseline,calibrated in [('delta_L','ridge0','pool100'),('delta_G','rkhs0','rkhs100'),('D','rkhs0','pool100')]:
            def vector(arm):
                return [next(x['r'] for x in family_arms if (x['dataset'],x['view'],x['q_percent'],x['arm'],x['family'])==('NAM_four_arm',site,10,arm,'N'+str(i))) for i in range(1,16)]
            b=np.asarray(vector(baseline));c=np.asarray(vector(calibrated))
            dispersion.append({'view':site,'contrast':contrast,'source_SD':float(np.std(b,ddof=1)),'calibrated_SD':float(np.std(c,ddof=1)),'source_calibrated_r':correlation(b,c),'Oldham_change_midpoint':correlation(c-b,(c+b)/2),'source_nonpositive':int(np.sum(b<=0)),'calibrated_nonpositive':int(np.sum(c<=0))})
    write_csv(out/'metrics.csv',rows)
    write_csv(out/'contrast_family_means.csv',families)
    write_csv(out/'contrast_summaries.csv',summaries)
    write_csv(out/'arm_family_means.csv',family_arms)
    write_csv(out/'arm_summaries.csv',arm_summaries)
    write_csv(out/'family_dispersion.csv',dispersion)
    undefined=sum(row['r'] is None for row in rows)
    report={'status':'complete' if undefined==0 else 'incomplete_undefined_r','frozen_cells':len(study['cells']),'metric_rows':len(rows),'reference_checks':checks,'reference_max_absolute_difference':worst,'undefined_r_rows':undefined,'new_fits':0,'new_tuning':0,'raw_input_refitting':False,'result_files':{p.name:digest(p.read_bytes()) for p in out.glob('*.csv')}}
    with (out/'RESULTS.json').open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report))
    if undefined:
        raise ValueError('Undefined r retained; complete reproduction not certified')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wei-xlsx',required=True)
    parser.add_argument('--irri-plotzip',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    execute(args)

if __name__=='__main__':
    main()
