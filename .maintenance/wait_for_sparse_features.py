"""Leave sparse AR series unpublished until a completed validation block exists."""
from pathlib import Path
import nbformat

p=Path('src/takehome/selection.py');s=p.read_text()
s=s.replace('weights=None, embargo: int = 1) -> PredictionSelection:',
            'weights=None, embargo: int = 1, allow_missing_history: bool = False) -> PredictionSelection:',1)
old='''            if not valid:
                raise ValueError(f'No eligible candidate in preceding validation fold {i-1}.')
            best = min(valid, key=lambda r: r['mse'])
            key = best['candidate']
            choices.append(key)
            selected[first:end] = arrays[key][first:end]
            decisions.append(dict(fold=i, validation_fold=i-1, candidate=key,
                validation_start=best['validation_start'], validation_stop=best['validation_stop'],
                validation_rows=best['n'], validation_mse=best['mse'],
                predict_start=first, predict_stop=end))'''
new='''            if not valid:
                if not allow_missing_history or any(r['n'] > 0 for r in previous):
                    raise ValueError(f'No eligible candidate in preceding validation fold {i-1}.')
                # No eligible historical examples is not a license for a static
                # alpha or a future-data choice. Publish no forecast in this block.
                best = previous[0]
                choices.append(None)
                decisions.append(dict(fold=i, validation_fold=i-1, candidate=None,
                    validation_start=best['validation_start'], validation_stop=best['validation_stop'],
                    validation_rows=0, validation_mse=np.nan, predict_start=first,
                    predict_stop=end, status='waiting_for_validation'))
            else:
                best = min(valid, key=lambda r: r['mse'])
                key = best['candidate']
                choices.append(key)
                selected[first:end] = arrays[key][first:end]
                decisions.append(dict(fold=i, validation_fold=i-1, candidate=key,
                    validation_start=best['validation_start'], validation_stop=best['validation_stop'],
                    validation_rows=best['n'], validation_mse=best['mse'],
                    predict_start=first, predict_stop=end, status='selected'))'''
assert old in s;s=s.replace(old,new,1)
s=s.replace("'predict_start', 'predict_stop']\n    score_columns", "'predict_start', 'predict_stop', 'status']\n    score_columns",1)
s=s.replace('''    are never used to select their own fold's candidate.
''','''    are never used to select their own fold's candidate. Sparse feature diagnostics
    can opt into allow_missing_history: a block with no past scoring examples
    leaves the next forecast missing. Invalid predictions on observed validation
    targets still raise. Return-model selection retains the strict default.
''',1)
s=s.replace("weights=ready.to_numpy(dtype=float), embargo=1)",
            "weights=ready.to_numpy(dtype=float), embargo=1, allow_missing_history=True)",1)
s=s.replace("selection=result.selection_, scores=result.validation_scores_, candidates=diagnostics)",
            "selection=result.selection_, scores=result.validation_scores_, choices=result.choices_,\n        candidates=diagnostics)",1)
p.write_text(s)

p=Path('notebooks/takehome.ipynb');nb=nbformat.read(p,4)
for c in nb.cells:
    tags=c.metadata.get('tags',[])
    if 'alpha_forecast_sweep' in tags:
        c.source=c.source.replace('ar_decisions, ar_fit_records, ar_error_rows = [], [], []',
                                 'ar_decisions, ar_fit_records, ar_error_rows = [], [], []\nar_fold_choices = {}')
        c.source=c.source.replace("            predictions[col] = prediction", "            predictions[col] = prediction\n            ar_fold_choices[col] = info['choices']")
        c.source += '''\nwaiting = ar_selection_table.status.eq('waiting_for_validation')
print('AR fold choices waiting for completed historical validation:', int(waiting.sum()))
display(ar_selection_table.loc[waiting, ['feature','fold','validation_rows','status']])
'''
    if 'alpha_forecast_controls' in tags:
        old="        ready=ready.where(pd.Series(rows >= alpha_first, index=df.index), False, axis=0)"
        new=old+'''
        for j,col in enumerate(X.columns):
            for fold,choice in zip(alpha_folds, ar_fold_choices[col]):
                if choice is None:
                    ready.iloc[fold['predict_start']:fold['predict_stop'], j] = False'''
        assert old in c.source;c.source=c.source.replace(old,new,1)
    if c.cell_type=='markdown' and c.source.startswith('#### Prior-test selection for AR(2)'):
        c.source += '''\n\nSparse or late-starting features may have no mature examples in an earlier validation block. Their next block remains **unavailable**, not filled with a fixed-alpha forecast or a winner chosen from future data. Selection resumes only after a completed preceding block supplies eligible examples. Pending choices and coverage are displayed; persistence/oracle controls use the same causal availability mask. All 200 return-model input columns are still retained.\n'''
    if c.cell_type=='code' and set(tags)&{'alpha_forecast_sweep','alpha_forecast_controls'}:
        c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
nbformat.write(nb,p)
p=Path('README.md');s=p.read_text()
s+='''\nSparse AR feature experiments leave a block unavailable when the previous test block contains no mature feature-target examples. They never substitute a static alpha or tune on the current block. The notebook reports pending choices and aligns the persistence/oracle control masks. This does not remove any raw or dszl column from the 200-input return models, whose regularization selection remains strict.\n'''
p.write_text(s)
p=Path('tests/test_prequential_selection.py');s=p.read_text()
old="    ns=dict(np=np,pd=pd,df=frame,x_cols=frame.columns,rows=np.arange(20),alpha_first=10,AR_MIN_TRAIN=3)"
new="""    ns=dict(np=np,pd=pd,df=frame,x_cols=frame.columns,rows=np.arange(20),alpha_first=10,AR_MIN_TRAIN=3,
            alpha_folds=[dict(predict_start=10,predict_stop=15),dict(predict_start=15,predict_stop=20)],
            ar_fold_choices={'x1':[None,.1],'x2':[.1,.1]})"""
assert old in s;s=s.replace(old,new,1)
s=s.replace("    assert_frame_equal(out.iloc[10:],frame.iloc[10:])", "    expected=frame.iloc[10:].copy();expected.iloc[:5,0]=np.nan\n    assert_frame_equal(out.iloc[10:],expected)",1)
if 'def test_feature_selection_waits_for_completed_validation_without_static_fallback' not in s:
    s+='''


def test_feature_selection_waits_for_completed_validation_without_static_fallback():
    mod=selection_module()
    y=np.zeros(40);y[10:20]=np.nan;y[20:30]=1
    preds={10.:np.zeros(40),.1:np.ones(40)}
    assert 'allow_missing_history' in __import__('inspect').signature(mod.select_previous_test).parameters
    result=mod.select_previous_test(preds,y,windows(),allow_missing_history=True)
    assert result.choices_==[None,None,.1]
    assert np.isnan(result.prediction_[:30]).all()
    assert_allclose(result.prediction_[30:],1)
    assert result.selection_.status.tolist()==['waiting_for_validation','selected']
    preds={a:np.full(40,np.nan) for a in (10.,.1)}
    with pytest.raises(ValueError,match='validation'):
        mod.select_previous_test(preds,np.ones(40),windows(),allow_missing_history=True)


def test_sparse_ar_feature_is_unavailable_until_previous_test_has_mature_examples():
    mod=selection_module();rng=np.random.default_rng(508)
    x=pd.Series(rng.normal(size=40),index=pd.date_range('2014',periods=40,freq='D'),name='late')
    x.iloc[:21]=np.nan
    pred,info=mod.forecast_alpha_selected(x,folds=windows(),alphas=[1.,.01],hl=5,min_train=3)
    assert pred.iloc[:30].isna().all()
    assert pred.iloc[30:].notna().all()
    assert info['choices'][:2]==[None,None]
    assert info['selection'].iloc[0].status=='waiting_for_validation'
    changed=x.copy();changed.iloc[30:]*=1000
    _,other=mod.forecast_alpha_selected(changed,folds=windows(),alphas=[1.,.01],hl=5,min_train=3)
    assert info['choices'][-1]==other['choices'][-1]
'''
p.write_text(s)
