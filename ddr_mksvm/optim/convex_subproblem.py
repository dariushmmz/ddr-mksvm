"""Auditable convex blocks for the specified DDR objective and legacy q=1 ablation."""
import warnings
import numpy as np
try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CVXPY_AVAILABLE = False

def _require_cvxpy():
    if not _CVXPY_AVAILABLE:
        raise ImportError("cvxpy is required; install compatible versions from requirements.txt")

if _CVXPY_AVAILABLE:
    _LP_SOLVER_CHAIN = (cp.HIGHS, cp.CLARABEL, cp.SCS)
    _SOCP_SOLVER_CHAIN = (cp.CLARABEL, cp.ECOS, cp.SCS)
    _SOLVER_KWARGS = {cp.CLARABEL: dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9),
                      cp.ECOS: dict(abstol=1e-9, reltol=1e-9, feastol=1e-9),
                      cp.SCS: dict(max_iters=50000, eps=1e-6)}

def _validate(K, y, nu, epsilon, L):
    K, y = np.asarray(K, float), np.asarray(y, float).reshape(-1)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != y.size:
        raise ValueError(f"expected K=(m,m), y=(m,); got K={K.shape}, y={y.shape}")
    if not np.isfinite(K).all() or not np.isfinite(y).all(): raise ValueError("K/y contain NaN or Inf")
    if set(np.unique(y)) != {-1.0, 1.0}: raise ValueError(f"labels must contain -1/+1; got {np.unique(y)}")
    if min(nu, epsilon, L) < 0: raise ValueError("nu, epsilon and L_theta_eta must be non-negative")
    scale=max(1.,float(np.abs(K).max()))
    if np.abs(K-K.T).max() > 1e-7*scale: raise ValueError("K is not symmetric")
    K=(K+K.T)/2
    mineig=float(np.linalg.eigvalsh(K).min())
    # Float32 Gram construction can create tiny negative eigenvalues even for
    # an exact Z.T@Z matrix. The solver's spectral factor clips only this
    # round-off; materially indefinite matrices are still rejected.
    if mineig < -1e-5*scale: raise ValueError(f"K is not PSD: min eigenvalue {mineig:.3e}")
    if mineig < 0:
        vals, vecs = np.linalg.eigh(K)
        K = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
        K = (K + K.T) / 2
    return K,y

def _psd_sqrt(M, eps_clip=0.0):
    vals, vecs=np.linalg.eigh((M+M.T)/2)
    return (vecs*np.sqrt(np.clip(vals,eps_clip,None)))@vecs.T

def _solve_with_fallback(problem, variables, is_socp, verbose=True):
    best=None
    for solver in (_SOCP_SOLVER_CHAIN if is_socp else _LP_SOLVER_CHAIN):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",UserWarning)
                problem.solve(solver=solver,**_SOLVER_KWARGS.get(solver,{}))
        except Exception: continue
        if any(v.value is None for v in variables.values()): continue
        st=problem.solver_stats
        snap=dict(status=problem.status,solver_name=st.solver_name,num_iters=st.num_iters,
                  solve_time=st.solve_time,primal_objective=float(problem.value),
                  **{n:(np.asarray(v.value) if np.ndim(v.value) else float(v.value)) for n,v in variables.items()})
        if problem.status=="optimal": return snap
        if best is None and problem.status=="optimal_inaccurate": best=snap
    if best is not None and verbose: warnings.warn("accepting optimal_inaccurate solver result")
    return best

def _line_search_b(M,u,y,gamma,xi):
    dxi=y*xi
    grid=np.linspace(gamma+1-np.max(-dxi),gamma-1+np.max(dxi),10000)
    counts=np.sum((-(M@u)[None,:]+y[None,:]*grid[:,None])>0,axis=1)
    i=int(np.argmin(counts))
    return float(grid[i] if counts[i]<len(y) else gamma)

def solution_diagnostics(K,y,solution,nu,epsilon=0.,L_theta_eta=0.,formulation="ddr_q2",
                         weight_tol=1e-8,score_tol=1e-8):
    K,y=_validate(K,y,nu,epsilon,L_theta_eta); M=np.outer(y,y)*K
    u=np.asarray(solution["u"],float); xi=np.maximum(np.asarray(solution["xi"],float),0)
    scores=K@(y*u)-float(solution["b"]); pred=np.where(scores>0,1.,-1.)
    w2=max(float(u@(M@u)),0.); wn=np.sqrt(w2)
    if formulation=="ddr_q2": reg,slack=nu*w2,float(xi.mean())
    elif formulation=="legacy_q1": reg,slack=float(np.abs(u).sum()),nu*float(xi.sum())
    else: raise ValueError(f"unknown formulation {formulation}")
    dro=epsilon*L_theta_eta*wn
    pv,pc=np.unique(pred,return_counts=True); _,yc=np.unique(y,return_counts=True)
    err=float(np.mean(pred!=y)); baseline=1-float(yc.max()/len(y))
    wc=wn<=weight_tol; sc=float(scores.std())<=score_tol; single=len(pv)==1
    sd=slack>=.99*max(reg+slack+dro,1e-15); be=abs(err-baseline)<=1/len(y)+1e-12
    return dict(objective=float(reg+slack+dro),regularization_term=float(reg),slack_term=float(slack),
      dro_term=float(dro),sum_xi=float(xi.sum()),xi_min=float(xi.min()),xi_max=float(xi.max()),
      xi_mean=float(xi.mean()),xi_std=float(xi.std()),xi_near_one=int(np.isclose(xi,1,atol=1e-6).sum()),
      u_l1=float(np.abs(u).sum()),u_l2=float(np.linalg.norm(u)),w_norm_H=float(wn),
      score_min=float(scores.min()),score_max=float(scores.max()),score_mean=float(scores.mean()),
      score_std=float(scores.std()),predicted_class_counts={int(v):int(c) for v,c in zip(pv,pc)},
      training_error=err,majority_baseline_error=baseline,weight_collapse=wc,score_collapse=sc,
      single_class_prediction=single,slack_dominated=sd,baseline_equivalent=be,
      degenerate=bool(wc or sc or single or (sd and be)))

def solve_svm_dro(K,y,nu,epsilon=0.,L_theta_eta=0.,formulation="ddr_q2"):
    """DDR uses mean(xi)+epsilon*L*||w||+nu*||w||^2; legacy is explicit."""
    _require_cvxpy(); K,y=_validate(K,y,nu,epsilon,L_theta_eta); m=len(y)
    M=np.outer(y,y)*K; H=_psd_sqrt(M)
    u,gamma,xi=cp.Variable(m),cp.Variable(),cp.Variable(m)
    constraints=[M@u-y*gamma+xi>=1,xi>=0]; use_dro=epsilon>0 and L_theta_eta>0
    if formulation=="ddr_q2":
        obj=cp.sum(xi)/m+nu*cp.sum_squares(H@u)
        if use_dro: obj+=epsilon*L_theta_eta*cp.norm(H@u,2)
        socp=True
    elif formulation=="legacy_q1":
        if use_dro: raise ValueError("legacy_q1 is deterministic; use ddr_q2 for Wasserstein DRO")
        s=cp.Variable(m); constraints += [u>=-s,u<=s,s>=0]; obj=cp.sum(s)+nu*cp.sum(xi); socp=False
    else: raise ValueError("formulation must be ddr_q2 or legacy_q1")
    problem=cp.Problem(cp.Minimize(obj),constraints)
    r=_solve_with_fallback(problem,dict(u=u,gamma=gamma,xi=xi),socp)
    if r is None:return None
    if not np.isfinite(r["u"]).all() or not np.isfinite(r["xi"]).all(): raise RuntimeError("non-finite solver variables")
    if np.min(r["xi"]) < -1e-5: raise RuntimeError("solver violated xi >= 0")
    u_value = np.asarray(r["u"], float)
    if formulation == "ddr_q2":
        # Kernel coefficients are non-unique when K is rank deficient. Remove
        # the null-space component (which changes neither w, margins nor the
        # objective) to avoid reporting enormous meaningless coefficients.
        vals, vecs = np.linalg.eigh(M)
        keep = vals > max(1.0, float(vals.max())) * 1e-10
        u_value = vecs[:, keep] @ (vecs[:, keep].T @ u_value)
    b=_line_search_b(M,u_value,y,r["gamma"],r["xi"])
    sol=dict(u=u_value,gamma=r["gamma"],b=b,xi=r["xi"],status=r["status"],
             solver_name=r["solver_name"],num_iters=r["num_iters"],solve_time=r["solve_time"],
             primal_objective=r["primal_objective"],used_dro=use_dro,formulation=formulation)
    sol["diagnostics"]=solution_diagnostics(K,y,sol,nu,epsilon,L_theta_eta,formulation)
    sol["training_error"]=sol["diagnostics"]["training_error"]
    return sol

def train_with_nu_search(K,y,nu_grid,epsilon=0.,L_theta_eta=0.,formulation="ddr_q2",verbose=True):
    best=None; candidates=[]
    for nu in np.asarray(nu_grid,float):
        sol=solve_svm_dro(K,y,float(nu),epsilon,L_theta_eta,formulation)
        if sol is None:continue
        d=sol["diagnostics"]; cand=dict(nu=float(nu),status=sol["status"],solver_name=sol["solver_name"],solution=sol,**d)
        candidates.append(cand)
        if verbose: print(f"    [SVM] nu={nu:.6g} status={sol['status']} error={d['training_error']:.6f} baseline={d['majority_baseline_error']:.6f} objective={d['objective']:.6g} reg={d['regularization_term']:.6g} slack={d['slack_term']:.6g} sum_xi={d['sum_xi']:.6g} ||w||_H={d['w_norm_H']:.6g} score_std={d['score_std']:.6g} preds={d['predicted_class_counts']} degenerate={d['degenerate']}")
        key=(d["training_error"],d["objective"],float(nu))
        if best is None or key<best[0]:best=(key,sol,float(nu))
    if best is None:return None
    result=best[1]; result["selected_nu"]=best[2]; result["_nu_candidates"]=candidates
    return result
