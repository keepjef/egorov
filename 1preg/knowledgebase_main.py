import json
import os
import pathlib
import copy
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

ROOT = pathlib.Path(os.getcwd())
DATA_FILE = ROOT / "knowledge_base_dat.json"

def _load_kb() -> Dict[str, Any]:
    if not DATA_FILE.exists():
        return {"facts": [], "rules": [], "goals": []}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

KB = _load_kb()

# === ДВИЖОК ===

def _eval_condition(cond: Any, memory: Dict[str, Any]) -> bool:
    # Здесь мы возвращаем строго bool, так как memory теперь не содержит None
    if isinstance(cond, dict) and "fact" in cond:
        fid = cond["fact"]
        expected = cond.get("eq", True)
        val = memory.get(fid, False) # Fallback to False just in case
        return val == expected

    if isinstance(cond, dict) and "all" in cond:
        return all(_eval_condition(c, memory) for c in cond.get("all") or [])

    if isinstance(cond, dict) and "any" in cond:
        return any(_eval_condition(c, memory) for c in cond.get("any") or [])

    if isinstance(cond, dict) and "not" in cond:
        return not _eval_condition(cond.get("not"), memory)

    return False

class Engine:
    def __init__(self, kb: dict):
        self.kb = kb
        # Инициализируем память. ВАЖНО: Если value is None -> ставим False.
        self.memory: Dict[str, bool] = {}
        for f in kb.get("facts", []):
            val = f.get("value")
            self.memory[f["id"]] = val if val is not None else False
            
        self.fired_rules: List[str] = []

    def _apply_actions(self, rule: dict) -> bool:
        changed = False
        for act in rule.get("then", []):
            fid = act.get("set")
            val = act.get("value", True)
            if self.memory.get(fid) != val:
                self.memory[fid] = val
                changed = True
        return changed

    def _activate_rule(self, rule_id: str, rule: dict, depth: int, max_depth: int):
        if depth > max_depth: return
        if rule_id in self.fired_rules: return

        # Теперь условие всегда возвращает True или False
        if _eval_condition(rule.get("if"), self.memory):
            if self._apply_actions(rule):
                self.fired_rules.append(rule_id)
                # Рестарт проверки правил при изменении памяти
                for r2 in self.kb.get("rules", []):
                    self._activate_rule(r2["id"], r2, depth + 1, max_depth)
            else:
                # Если условие верно, но память не поменялась (уже было True), помечаем как сработавшее
                self.fired_rules.append(rule_id)

    def run(self, max_depth=100):
        for r in self.kb.get("rules", []):
            self._activate_rule(r["id"], r, 0, max_depth)
        return self._result()

    def _result(self) -> Dict[str, Any]:
        goals = {}
        goal_ids = set(self.kb.get("goals", []))
        for item in self.kb.get("facts", []):
            if item["id"] in goal_ids:
                goals[item["label"]] = self.memory.get(item["id"])
        
        return {
            "memory": self.memory,
            "goals": goals,
            "fired_rules": self.fired_rules
        }

def merge_inputs(kb: dict, inputs: Dict[str, Any]) -> dict:
    kb2 = copy.deepcopy(kb)
    # Проставляем пользовательский ввод
    # Все, что не пришло в inputs, останется None и превратится в False при инициализации Engine
    for f in kb2.get("facts", []):
        if f["id"] in inputs:
            f["value"] = inputs[f["id"]]
        else:
            # Принудительно ставим False для всего, что не выбрано
            # Кроме целей (они вычисляются)
            if f["id"] not in kb2.get("goals", []):
                f["value"] = False
    return kb2

class InferenceRequest(BaseModel):
    inputs: Dict[str, bool] = Field(default_factory=dict)

class InferenceResponse(BaseModel):
    memory: Dict[str, Any]
    goals: Dict[str, bool]
    fired_rules: List[str]

app = FastAPI()

@app.get("/kb")
def get_input_facts():
    # Отдаем только вопросы (не цели)
    goal_ids = set(KB.get("goals", []))
    facts = [f for f in KB.get("facts", []) if f["id"] not in goal_ids]
    return {"facts": facts}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    kb = merge_inputs(KB, req.inputs)
    eng = Engine(kb)
    return eng.run()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("knowledgebase_main:app", host="127.0.0.1", port=8000, reload=True)