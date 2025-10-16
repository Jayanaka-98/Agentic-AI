jac2_url = "http://localhost:8000"
user_token = ""

import random
import string
import json
import os
import datetime
import requests
from pathlib import Path

jac2_session = requests.Session()

def generate_password(length=12):
    chars = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(chars) for _ in range(length))
    return password

def manage_user_credentials(username, existing_password="", credentials_file='user_credentials.json'):
    # Create credentials file if it doesn't exist
    if not os.path.exists(credentials_file):
        with open(credentials_file, 'w') as f:
            json.dump({}, f)
    
    # Load existing credentials
    with open(credentials_file, 'r') as f:
        credentials = json.load(f)
    
    if existing_password:  # If password is provided, use it
        password = existing_password
    elif username not in credentials or 'password' not in credentials[username]:  # Generate new if needed
        password = generate_password()
    else:  # Use existing password from credentials
        password = credentials[username]['password']
    
    # Update credentials in JSON
    credentials[username] = {
        'password': password,
        'created_at': str(datetime.datetime.now())
    }
    
    # Save updated credentials
    with open(credentials_file, 'w') as f:
        json.dump(credentials, f, indent=4)
    
    return password

def login_user(username, password):
    # Placeholder for actual login logic
    
    email = username
    pw = password
    res = jac2_session.post(
        f"{jac2_url}/user/register",
        json={"email": user, "password": pw, "is_activated": True},
    )

    res = jac2_session.post(
        f"{jac2_url}/user/login/",
        json={"email": user, "password": pw},
    )
    jac2_session.headers.update({"Content-Type": "application/json"})
    # print(res.json())
    user_token = res.json()['token']
    jac2_session.headers.update({"Authorization": f"bearer {user_token}"})
    jac2_session.post(f"{jac2_url}/walker/init_user", json={})
    # load_data(username)
    

def load_data(username):
    localpart = username.split("@", 1)[0]
    
    # --- read from ./raw/ instead of ./test_data/ ---
    src_path = Path(f'./raw/{localpart}.json')

    # --- output folders remain the same ---
    test_data_dir = Path("./test_data")
    qa_dir = Path("./qa")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    test_data_out_path = test_data_dir / f"{localpart}.json"
    qa_out_path = qa_dir / f"{localpart}.json"

    with src_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
        
    if "memories" in data:
        user_id = data.get("user_id", localpart)
        memories = data.get("memories", [])
        qa_pairs = data.get("qa_pairs", [])

        reports = []
        for m in memories:
            memory_id_val = m.get("memory_id", "")
            summary_val   = m.get("summary", "")
            when_val      = m.get("when", "")
            who_val       = m.get("who", [])
            where_val     = m.get("where", [])
            what_val      = m.get("what", "")

            reports.append({
                "id": memory_id_val,
                "context": {
                    "memory_id": memory_id_val,
                    "summary": summary_val,
                    "comments_summary": "",
                    "when": [when_val] if isinstance(when_val, str) else when_val,
                    "who": who_val,
                    "where": where_val,
                    "what": [what_val] if isinstance(what_val, str) else what_val,
                    "natural_when": "",
                    "emotion": "",
                    "created_at": "",
                    "updated_at": "",
                    "image_urls": [],
                    "new_image_format": "",
                    "shared_with": [],
                    "conversation": [],
                    "session_id": "",
                    "draft": False
                }
            })

        migrate_payload = {"status": 200, "reports": reports}

        # --- Save migrated test_data ---
        with test_data_out_path.open("w", encoding="utf-8") as f:
            json.dump(migrate_payload, f, indent=2, ensure_ascii=False)

        # --- Save QA to qa/<localpart>.json ---
        qa_payload = {"user_id": user_id, "qa_pairs": qa_pairs}
        with qa_out_path.open("w", encoding="utf-8") as f:
            json.dump(qa_payload, f, indent=2, ensure_ascii=False)

        profile_json = migrate_payload
    else:
        # Already migrated
        profile_json = data

    # ---- migrate to backend ----
    res = jac2_session.post(
        f"{jac2_url}/walker/migrate_profile_data",
        json={"json_file_content": profile_json["reports"]},
    )
    try:
        print(res.json())
    except Exception:
        print({"status_code": res.status_code, "text": res.text})

    if res.status_code == 200:
        print(f"Profile migrated for {username}")

    res = jac2_session.post(f"{jac2_url}/walker/list_memories", json={})
    try:
        print(res.json())
    except Exception:
        print({"status_code": res.status_code, "text": res.text})
    
# Example usage:
if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("mode", nargs="?", default="run", choices=["migrate", "run"],
                        help="migrate: load_data once per user and exit; run: query QA only (no load_data)")
    parser.add_argument("--exp", default="mtp", choices=["mtp", "mtp_with_sem", "pe"],
                        help="experiment type controls result output directory")
    args = parser.parse_args()

    MODE = args.mode  # "migrate" or "run"
    EXP = args.exp

    # prepare passwords for all users
    users = {f"user_{i:03d}@jaseci.org": "" for i in range(1, 41)}
    for u, pw in users.items():
        users[u] = manage_user_credentials(u, pw)

    if MODE == "migrate":
        # --- MIGRATION PHASE: call load_data exactly once per user, no querying ---
        for email in sorted(users.keys()):
            user = email
            try:
                login_user(email, users[email])
                load_data(email)
                print(f"[migrate] migrated memories for {email}")
            except Exception as e:
                print({"user": email, "error": f"migration_failed: {repr(e)}"})
        print("[migrate] done.")
        raise SystemExit(0)

    # --- RUN PHASE ---
    # dynamically choose result folder based on experiment type
    base_results_dir = Path("results")
    exp_results_dir = base_results_dir / EXP
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    all_out_path = exp_results_dir / "all_users_answers.jsonl"

    with open(all_out_path, "w", encoding="utf-8") as all_out:
        for email in sorted(users.keys()):
            user = email  # IMPORTANT for login_user()
            try:
                login_user(email, users[email])
            except Exception as e:
                err = {"user": email, "error": f"login_failed: {repr(e)}"}
                print(err)
                all_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                continue

            localpart = email.split("@", 1)[0]
            qa_path = f"./qa/{localpart}.json"
            results_path = exp_results_dir / f"{localpart}_answers.jsonl"

            if not os.path.exists(qa_path):
                warn = {"user": email, "warning": f"qa_file_missing: {qa_path}"}
                print(warn)
                all_out.write(json.dumps(warn, ensure_ascii=False) + "\n")
                continue

            with open(qa_path, "r", encoding="utf-8") as f:
                qa_payload = json.load(f)

            qa_pairs = qa_payload.get("qa_pairs", [])
            with open(results_path, "w", encoding="utf-8") as out:
                for qa in qa_pairs:
                    qid = qa.get("qid")
                    question = qa.get("question", "")
                    gold = qa.get("answer_gold", "")
                    evidence = qa.get("evidence", [])

                    resp = jac2_session.post(
                        f"{jac2_url}/walker/search_memories",
                        json={"query": question},
                    )
                    try:
                        resp_json = resp.json()
                    except Exception:
                        resp_json = {"status_code": resp.status_code, "text": resp.text}

                    # extract ONLY memory_id(s) from arbitrarily nested "reports"
                    memory_ids = []
                    if isinstance(resp_json, dict):
                        queue = list(resp_json.get("reports", []))
                        while queue:
                            item = queue.pop(0)
                            if isinstance(item, list):
                                queue.extend(item)
                            elif isinstance(item, dict):
                                mem = item.get("memory", item)
                                mid = mem.get("memory_id") or mem.get("id")
                                if isinstance(mid, str) and mid:
                                    memory_ids.append(mid)

                    record = {
                        "user": email,
                        "qid": qid,
                        "question": question,
                        "answer": evidence,
                        "output": memory_ids,
                    }
                    print(record)
                    line = json.dumps(record, ensure_ascii=False)
                    out.write(line + "\n")
                    all_out.write(line + "\n")

            print(f"[done] {email} -> {results_path}")

    print(f"[all done] wrote aggregate: {all_out_path}")