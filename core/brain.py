# Brain coordinates the LLM with memory and returns a plan and reply

import json

class Brain:
    def __init__(self, llm, memory, system_prompt, planner_schema, logger=None):
        # Save collaborators
        self.llm = llm
        self.mem = memory
        self.system_prompt = system_prompt
        self.schema = planner_schema
        self.log = logger or (lambda s: None)

    def plan_dryrun(self, user_text:str):
        # Return a trivial plan without calling tools
        plan = []
        if user_text.strip().startswith("read file "):
            plan.append("Read the file")
        elif user_text.strip() in ("tree", "ls"):
            plan.append("List directory")
        else:
            plan.append("Reply to the user")
        return {"plan": plan}

    def step(self, user_text:str):
        # Compose messages for the LLM
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text}
        ]
        # Query the model
        reply = self.llm.chat(msgs).strip()
        # Store memory
        self.mem.add("user", user_text)
        self.mem.add("assistant", reply)
        # Simple plan extraction
        plan = []
        if "plan:" in reply.lower():
            plan.append("Execute plan from model")
        elif user_text.strip().startswith("read file"):
            plan.append("Read the file")
        elif user_text.strip() in ("tree", "ls"):
            plan.append("List directory")
        # Return structured result
        return {"plan": plan, "reply": reply}
