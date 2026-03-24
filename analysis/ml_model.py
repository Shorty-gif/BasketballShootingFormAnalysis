"""
analysis/ml_model.py  —  XGBoost + k-NN with expanded player database (20+ players)
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pickle
    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False

ROOT        = Path(__file__).parent.parent
MODELS_DIR  = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "prediction_results"
XGB_PATH    = MODELS_DIR / "xgb_form_scorer.json"
SCALER_PATH = MODELS_DIR / "knn_scaler.pkl"

# FIX #5: Expanded to 22 pro players with diverse styles
PRO_PLAYER_DB = [
    # Pure shooters
    {"name":"Stephen Curry",    "team":"Golden State Warriors", "style":"high arc quick release",
     "elbow_angle":88,  "knee_angle":118, "shoulder_angle":68, "wrist_elbow_v":12, "hip_knee_align":0.05,
     "notes":"One-motion shot, extremely consistent elbow tuck"},
    {"name":"Klay Thompson",    "team":"Golden State Warriors", "style":"catch and shoot",
     "elbow_angle":85,  "knee_angle":122, "shoulder_angle":64, "wrist_elbow_v":10, "hip_knee_align":0.06,
     "notes":"Textbook catch-and-shoot, minimal dip, quick release"},
    {"name":"Ray Allen",        "team":"Retired", "style":"fundamentally pure",
     "elbow_angle":90,  "knee_angle":115, "shoulder_angle":62, "wrist_elbow_v":8,  "hip_knee_align":0.04,
     "notes":"Greatest shooting form ever — elbow directly under ball"},
    {"name":"Larry Bird",       "team":"Retired", "style":"high arc set shot",
     "elbow_angle":87,  "knee_angle":120, "shoulder_angle":65, "wrist_elbow_v":9,  "hip_knee_align":0.05,
     "notes":"Perfect vertical elbow alignment, high arc release"},
    {"name":"Reggie Miller",    "team":"Retired", "style":"off balance specialist",
     "elbow_angle":95,  "knee_angle":110, "shoulder_angle":70, "wrist_elbow_v":18, "hip_knee_align":0.12,
     "notes":"Unorthodox hip alignment but perfect elbow tracking"},
    {"name":"Damian Lillard",   "team":"Milwaukee Bucks", "style":"deep range pull-up",
     "elbow_angle":86,  "knee_angle":125, "shoulder_angle":66, "wrist_elbow_v":11, "hip_knee_align":0.07,
     "notes":"Powerful leg drive, consistent elbow tuck on pull-ups"},
    {"name":"Steph Curry (off dribble)", "team":"Golden State Warriors", "style":"off dribble quick",
     "elbow_angle":91,  "knee_angle":112, "shoulder_angle":70, "wrist_elbow_v":14, "hip_knee_align":0.09,
     "notes":"Even off dribble maintains elbow discipline"},
    {"name":"Trae Young",       "team":"Atlanta Hawks", "style":"floater specialist",
     "elbow_angle":83,  "knee_angle":108, "shoulder_angle":72, "wrist_elbow_v":13, "hip_knee_align":0.08,
     "notes":"High release point, excellent wrist snap"},
    {"name":"Devin Booker",     "team":"Phoenix Suns", "style":"mid-range fundamentals",
     "elbow_angle":89,  "knee_angle":120, "shoulder_angle":63, "wrist_elbow_v":10, "hip_knee_align":0.05,
     "notes":"Kobe-inspired mid-range mechanics, excellent follow-through"},
    # Power / athletic shooters
    {"name":"Kevin Durant",     "team":"Phoenix Suns", "style":"high release height",
     "elbow_angle":92,  "knee_angle":128, "shoulder_angle":72, "wrist_elbow_v":14, "hip_knee_align":0.04,
     "notes":"Elite shoulder elevation, releases above defenders"},
    {"name":"LeBron James",     "team":"Los Angeles Lakers", "style":"power form",
     "elbow_angle":100, "knee_angle":132, "shoulder_angle":60, "wrist_elbow_v":16, "hip_knee_align":0.07,
     "notes":"More elbow flare than optimal — compensates with power"},
    {"name":"Giannis Antetokounmpo","team":"Milwaukee Bucks", "style":"improved shooter",
     "elbow_angle":105, "knee_angle":125, "shoulder_angle":55, "wrist_elbow_v":20, "hip_knee_align":0.10,
     "notes":"Developed shooter — elbow still slightly wide but improving"},
    {"name":"Joel Embiid",      "team":"Philadelphia 76ers", "style":"big man shooter",
     "elbow_angle":93,  "knee_angle":118, "shoulder_angle":65, "wrist_elbow_v":15, "hip_knee_align":0.06,
     "notes":"Surprisingly clean mechanics for a center, good knee bend"},
    {"name":"Nikola Jokic",     "team":"Denver Nuggets", "style":"unorthodox but effective",
     "elbow_angle":108, "knee_angle":114, "shoulder_angle":58, "wrist_elbow_v":22, "hip_knee_align":0.13,
     "notes":"Unconventional push shot but high release point works"},
    # Fadeaways / specialised
    {"name":"Dirk Nowitzki",    "team":"Retired", "style":"one legged fadeaway",
     "elbow_angle":98,  "knee_angle":145, "shoulder_angle":78, "wrist_elbow_v":22, "hip_knee_align":0.18,
     "notes":"Higher elbow and knee extension — unique one-legged fadeaway"},
    {"name":"Kobe Bryant",      "team":"Retired", "style":"mid-range assassin",
     "elbow_angle":88,  "knee_angle":118, "shoulder_angle":67, "wrist_elbow_v":11, "hip_knee_align":0.05,
     "notes":"Perfect mechanics, elite elbow tuck and follow-through"},
    {"name":"Michael Jordan",   "team":"Retired", "style":"mid-range perfection",
     "elbow_angle":90,  "knee_angle":120, "shoulder_angle":65, "wrist_elbow_v":9,  "hip_knee_align":0.04,
     "notes":"Legendary form — balance, elbow alignment, and wrist snap"},
    {"name":"Paul Pierce",      "team":"Retired", "style":"shot fake and rise",
     "elbow_angle":96,  "knee_angle":116, "shoulder_angle":62, "wrist_elbow_v":17, "hip_knee_align":0.08,
     "notes":"Slightly unorthodox but effective elbow position"},
    # Modern guards
    {"name":"Jayson Tatum",     "team":"Boston Celtics", "style":"versatile scorer",
     "elbow_angle":87,  "knee_angle":122, "shoulder_angle":66, "wrist_elbow_v":12, "hip_knee_align":0.06,
     "notes":"Clean mechanics, good knee drive, consistent release"},
    {"name":"Luka Doncic",      "team":"Dallas Mavericks", "style":"step back specialist",
     "elbow_angle":94,  "knee_angle":115, "shoulder_angle":64, "wrist_elbow_v":16, "hip_knee_align":0.09,
     "notes":"Natural shooter, slightly wide elbow on step-backs"},
    {"name":"Anthony Edwards",  "team":"Minnesota Timberwolves", "style":"explosive scorer",
     "elbow_angle":97,  "knee_angle":126, "shoulder_angle":63, "wrist_elbow_v":14, "hip_knee_align":0.07,
     "notes":"Powerful through the ball, strong leg drive"},
    {"name":"Zach LaVine",      "team":"Chicago Bulls", "style":"athletic shooter",
     "elbow_angle":86,  "knee_angle":121, "shoulder_angle":67, "wrist_elbow_v":11, "hip_knee_align":0.05,
     "notes":"Smooth mechanics, elite athleticism enhances release height"},
]

FEATURE_COLS = ["elbow_angle","knee_angle","shoulder_angle","wrist_elbow_v","hip_knee_align"]


def _player_features(p: Dict) -> np.ndarray:
    return np.array([p["elbow_angle"],p["knee_angle"],p["shoulder_angle"],
                     p["wrist_elbow_v"],p["hip_knee_align"]], dtype=np.float32)


def metrics_to_features(m: Dict) -> Optional[np.ndarray]:
    try:
        return np.array([
            float(m["elbow_angle"]),
            float(m["knee_angle"]),
            float(m.get("shoulder_angle", 60)),
            float(m.get("wrist_elbow_vertical", m.get("wrist_elbow_v", 15))),
            float(m.get("hip_knee_alignment", m.get("hip_knee_align", 0.08))),
        ], dtype=np.float32)
    except (KeyError,TypeError,ValueError):
        return None


PHASE_IDEALS = {
    "loading":        {"elbow":(85,115), "knee":(95,125),  "shoulder":(30,55)},
    "set":            {"elbow":(80,100), "knee":(100,130), "shoulder":(48,72)},
    "release":        {"elbow":(78,108), "knee":(125,175), "shoulder":(58,88)},
    "follow_through": {"elbow":(138,180),"knee":(148,180), "shoulder":(62,90)},
    "not_shooting":   {"elbow":(60,120), "knee":(155,180), "shoulder":(0,30)},
}


def _rule_score(features: np.ndarray, phase: str = "set") -> float:
    elbow,knee,shoulder,wrist_v,hip = features
    ideals = PHASE_IDEALS.get(phase, PHASE_IDEALS["set"])
    def s(val,imin,imax,am,ax):
        if imin<=val<=imax:
            c=(imin+imax)/2; sp=(imax-imin)/2+1e-8
            return min(100,85+15*max(0,1-abs(val-c)/sp))
        elif am<=val<=ax:
            return 50+35*((val-am)/(imin-am+1e-8) if val<imin else (ax-val)/(ax-imax+1e-8))
        m=25.0
        return max(0,50*(val-(am-m))/m if val<am else 50*((ax+m)-val)/m)
    ei,ki,si = ideals["elbow"],ideals["knee"],ideals["shoulder"]
    scores = {
        "elbow":    s(elbow,   ei[0],ei[1],ei[0]-15,ei[1]+20),
        "knee":     s(knee,    ki[0],ki[1],ki[0]-15,ki[1]+25),
        "shoulder": s(shoulder,si[0],si[1],si[0]-15,si[1]+15),
        "wrist_v":  max(0,100-wrist_v*2.0),
        "hip":      max(0,100-hip*350),
    }
    return float(sum(v*w for v,w in zip(scores.values(),[0.30,0.20,0.20,0.15,0.15])))


def generate_training_data(n: int = 6000) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X, y = [], []
    for phase,ideals in PHASE_IDEALS.items():
        ei,ki,si = ideals["elbow"],ideals["knee"],ideals["shoulder"]
        centre = np.array([(ei[0]+ei[1])/2,(ki[0]+ki[1])/2,(si[0]+si[1])/2,12.0,0.06],dtype=np.float32)
        ns = n // (len(PHASE_IDEALS)*3)
        for s in centre + np.random.normal(0,1,(ns,5))*[8,12,8,4,0.02]:
            X.append(s); y.append(_rule_score(s,phase))
    for player in PRO_PLAYER_DB:
        pf = _player_features(player)
        for phase in PHASE_IDEALS:
            ns = n // (len(PRO_PLAYER_DB)*len(PHASE_IDEALS)*2)
            for s in pf + np.random.normal(0,1,(max(1,ns),5))*[5,8,5,3,0.015]:
                X.append(s); y.append(_rule_score(s,phase))
    for bp,phase in [([145,165,45,40,0.25],"set"),([60,170,25,50,0.30],"loading"),
                     ([130,155,90,35,0.20],"set"),([60,168,20,55,0.22],"loading")]:
        feats = np.array(bp,dtype=np.float32)
        for s in feats + np.random.normal(0,1,(n//(4*4),5))*[10,15,10,6,0.03]:
            X.append(s); y.append(_rule_score(s,phase))
    return np.array(X,dtype=np.float32), np.clip(np.array(y,dtype=np.float32),0,100)


class XGBoostFormScorer:
    def __init__(self): self.model = None

    def train(self,X,y) -> Dict:
        if not XGB_AVAILABLE: return {"error":"xgboost not installed"}
        X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
        self.model = xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.04,
            subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,
            random_state=42,verbosity=0)
        self.model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
        preds = self.model.predict(X_val)
        imp   = dict(zip(FEATURE_COLS,[round(float(x),4) for x in self.model.feature_importances_]))
        MODELS_DIR.mkdir(parents=True,exist_ok=True)
        self.model.save_model(str(XGB_PATH))
        return {"val_mae":round(float(np.mean(np.abs(preds-y_val))),2),
                "feature_importance":imp}

    def load(self) -> bool:
        if not XGB_AVAILABLE or not XGB_PATH.exists(): return False
        self.model = xgb.XGBRegressor(); self.model.load_model(str(XGB_PATH)); return True

    def predict(self, features: np.ndarray, phase: str = "set") -> Dict:
        rule = _rule_score(features, phase)
        if self.model is not None:
            score = round(float(np.clip(self.model.predict(features.reshape(1,-1))[0],0,100))*0.6+rule*0.4,1)
            source = "xgboost+rules"
        else:
            score = round(rule,1); source = "rules_only"
        imp  = dict(zip(FEATURE_COLS,self.model.feature_importances_)) if self.model else {}
        top  = max(imp,key=imp.get).replace("_"," ") if imp else "elbow angle"
        if score>=85: exp="Excellent form — all angles ideal for this phase"
        elif score>=70: exp="Good form — main area to refine: "+top
        elif score>=50: exp="Average form — focus on: "+top
        else: exp="Needs work — primary issue: "+top
        return {"xgb_score":score,"explanation":exp,"top_factor":top,"source":source}


class KNNStyleMatcher:
    def __init__(self,k=3): self.k=k; self.X_db=None; self.scaler=None; self._ready=False

    def fit(self) -> bool:
        if not SKL_AVAILABLE: return False
        X_raw = np.array([_player_features(p) for p in PRO_PLAYER_DB])
        self.scaler = StandardScaler(); self.X_db = self.scaler.fit_transform(X_raw)
        self._ready = True
        MODELS_DIR.mkdir(parents=True,exist_ok=True)
        with open(SCALER_PATH,"wb") as f: pickle.dump(self.scaler,f)
        return True

    def load(self) -> bool:
        if SCALER_PATH.exists():
            try:
                with open(SCALER_PATH,"rb") as f: self.scaler=pickle.load(f)
                self.X_db = self.scaler.transform(np.array([_player_features(p) for p in PRO_PLAYER_DB]))
                self._ready=True; return True
            except Exception: pass
        return self.fit()

    def match(self, features: np.ndarray, phase: str = "set") -> Dict:
        if not self._ready: self.load()

        # Phase correction: convert measured angles to set-point equivalent
        # so follow-through / release angles compare fairly against pro DB
        # (which stores set-point angles for all players)
        PHASE_CORR = {
            "follow_through": {"elbow": -55, "knee": -42},
            "release":        {"elbow": -18, "knee": -22},
            "loading":        {"elbow":  +5, "knee": +18},
            "set":            {"elbow":   0, "knee":   0},
            "not_shooting":   {"elbow":   0, "knee":   0},
        }
        corr = PHASE_CORR.get(phase, PHASE_CORR["set"])
        f_corrected = features.copy()
        f_corrected[0] = np.clip(features[0] + corr["elbow"], 60, 130)
        f_corrected[1] = np.clip(features[1] + corr["knee"],  85, 160)

        if self.scaler is not None:
            q  = self.scaler.transform(f_corrected.reshape(1,-1))[0]
            db = self.X_db
        else:
            q  = f_corrected
            db = np.array([_player_features(p) for p in PRO_PLAYER_DB])

        dists   = np.linalg.norm(db - q, axis=1)
        top_idx = np.argsort(dists)[:min(self.k, len(PRO_PLAYER_DB))]

        # Calibrate decay constant from median distance in DB
        median_dist = float(np.median(dists))
        decay = max(2.0, median_dist / 2.0)

        matches = []
        for rank, idx in enumerate(top_idx):
            player = PRO_PLAYER_DB[idx]
            dist   = float(dists[idx])

            # Exponential decay similarity — always non-zero, meaningful at any distance
            similarity = round(100.0 * float(np.exp(-dist / decay)), 1)

            # Show diffs against ORIGINAL angles (not corrected) for honest advice
            pf    = _player_features(player)
            diffs = features - pf
            advice = []
            for fn, diff, thr, unit in zip(
                    ["elbow", "knee", "shoulder", "wrist tilt", "hip align"],
                    diffs, [8, 12, 8, 6, 0.05],
                    ["deg", "deg", "deg", "deg", ""]):
                if abs(diff) > thr:
                    advice.append(fn + " is " + f"{abs(diff):.1f}" + unit +
                                  (" higher" if diff > 0 else " lower") +
                                  " than " + player["name"] + " at set point")

            matches.append({
                "rank":       rank + 1,
                "name":       player["name"],
                "team":       player["team"],
                "style":      player["style"],
                "similarity": similarity,
                "distance":   round(dist, 3),
                "notes":      player["notes"],
                "advice":     advice[:2],
            })

        best    = matches[0] if matches else {}
        summary = ("Your form is most similar to " + best["name"] +
                   " (" + str(best["similarity"]) + "% match). " +
                   best["notes"]) if best else ""
        return {"matches": matches, "best_match": best.get("name", ""), "summary": summary}


class BasketballMLPredictor:
    def __init__(self):
        self.xgb=XGBoostFormScorer(); self.knn=KNNStyleMatcher(k=3); self._ready=False

    def _ensure_ready(self):
        if not self._ready: self.xgb.load(); self.knn.load(); self._ready=True

    def predict(self, metrics: Dict) -> Dict:
        features = metrics_to_features(metrics)
        if features is None:
            return {"xgb_score":None,"xgb_explanation":"Feature extraction failed",
                    "knn_matches":[],"knn_summary":"","knn_best_match":""}
        self._ensure_ready()
        phase      = metrics.get("phase","set")
        xgb_result = self.xgb.predict(features, phase)
        knn_result = self.knn.match(features, phase)
        return {"xgb_score":xgb_result["xgb_score"],"xgb_explanation":xgb_result["explanation"],
                "xgb_top_factor":xgb_result["top_factor"],"knn_matches":knn_result["matches"],
                "knn_summary":knn_result["summary"],"knn_best_match":knn_result["best_match"]}


def train_all(data_path=None):
    print("\nGenerating training data for all phases...")
    X,y = generate_training_data(n=7000)
    if data_path and Path(data_path).exists():
        with open(data_path) as f: data=json.load(f)
        for m in data.get("per_image",[]):
            feats=metrics_to_features(m); score=m.get("overall_score")
            if feats is not None and score is not None:
                X=np.vstack([X,feats]); y=np.append(y,float(score))
    print("Training XGBoost on "+str(len(X))+" samples...")
    scorer=XGBoostFormScorer(); r=scorer.train(X,y)
    if "error" not in r:
        print("MAE: "+str(r["val_mae"]))
        for k,v in sorted(r["feature_importance"].items(),key=lambda x:x[1],reverse=True):
            print("  "+k.ljust(20)+"█"*int(v*40)+"  "+str(round(v*100,1))+"%")
    print("\nFitting k-NN on "+str(len(PRO_PLAYER_DB))+" pro players...")
    KNNStyleMatcher().fit()
    print("\nDemo (Curry-like):")
    pred=BasketballMLPredictor()
    pred._ready=False
    r=pred.predict({"elbow_angle":88,"knee_angle":118,"shoulder_angle":68,
                    "wrist_elbow_vertical":12,"hip_knee_alignment":0.05,"phase":"set"})
    print("  Score: "+str(r["xgb_score"]))
    for m in r["knn_matches"]: print("  "+str(m["rank"])+". "+m["name"]+" — "+str(m["similarity"])+"%")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--demo",action="store_true")
    parser.add_argument("--data",type=str,default=str(METRICS_DIR/"metrics_summary.json"))
    args=parser.parse_args()
    if args.train: train_all(args.data)
    elif args.demo:
        p=BasketballMLPredictor()
        r=p.predict({"elbow_angle":88,"knee_angle":118,"shoulder_angle":68,
                     "wrist_elbow_vertical":12,"hip_knee_alignment":0.05,"phase":"set"})
        print("Score:",r["xgb_score"])
        for m in r["knn_matches"]: print(str(m["rank"])+". "+m["name"]+" — "+str(m["similarity"])+"%")
    else: print("Usage: python analysis/ml_model.py --train")