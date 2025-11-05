import from langchain_core.messages { HumanMessage, AIMessage, BaseMessage }
import from states { AppState }
import from graph { build_graph }
import from dotenv { load_dotenv }
import from memory { Memory, Session }
import os;

walker task_manager {
    has utterance: str = "";
    has session_id: str = "";
    has response: str = "";

    obj __specs__ {
        static has auth: bool = False;
    }
    can call_build_graph with `root entry {
        memory_list = [root --> (`?Memory)];
        if not memory_list {
            memory_list = root ++> Memory();
        }
        memory = memory_list[0];
        if not self.session_id {
            session_list = memory ++> Session();
            self.session = session_list[0];
        } else {
            self.session = &(self.session_id);
        }
        
        app = build_graph();
        carry_keys: dict = {"email_to", "email_subject", "email_content"};
        carry_forward = {k: v for (k, v) in self.session.last_state.items() if k in carry_keys};

        state: AppState = {"utterance": self.utterance, "history": self.session.history, **carry_forward};
        out = app.invoke(state);
        reply = out.get("response", "");
        print("\nAssistant:", reply, "\n");
        self.response = reply;
        report {
            "response": reply,
            "session_id": jid(self.session),
        };

        self.session.add_history(f"Human: {self.utterance}\nAI: {reply}");
        self.session.last_state = out;
    }
}

with entry {
    load_dotenv();
    import json;
    import datetime;
    with open("generalized_synthetic_agent_data.json", "r") as f {
        data = json.load(f);
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S");
    for type in data.keys() {
        for utterance in data[type] {
            try {
            root spawn (x := task_manager(utterance=utterance));
            } except Exception as e {
                print(f"Error processing utterance '{utterance}': {e}");
            }
            record = {
                "type": type,
                "utterance": x.utterance,
                "response": x.response
            };
            with open(f"taskman_{timestamp}.jsonl", "a") as out_f {
                out_f.write(json.dumps(record) + "\n");
            }
        }
    }
}