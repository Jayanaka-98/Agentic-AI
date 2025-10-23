jac2_url = "http://localhost:8000"
user_token = ""

import random, string, json, os, datetime, requests
from pathlib import Path

jac2_session = requests.Session()

def generate_password(length=12):
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(chars) for _ in range(length))

def manage_user_credentials(username, existing_password="", credentials_file='user_credentials.json'):
    if not os.path.exists(credentials_file):
        with open(credentials_file, 'w') as f:
            json.dump({}, f)

    with open(credentials_file, 'r') as f:
        credentials = json.load(f)

    if existing_password:
        password = existing_password
    elif username not in credentials or 'password' not in credentials[username]:
        password = generate_password()
    else:
        password = credentials[username]['password']

    credentials[username] = {
        'password': password,
        'created_at': str(datetime.datetime.now())
    }

    with open(credentials_file, 'w') as f:
        json.dump(credentials, f, indent=4)

    return password


def login_user(username, password):
    email = username
    pw = password

    # --- Register user if not exists ---
    try:
        jac2_session.post(
            f"{jac2_url}/user/register",
            json={"email": email, "password": pw, "is_activated": True},
        )
    except Exception as e:
        print(f"[warn] registration failed for {email}: {e}")

    # --- Login ---
    res = jac2_session.post(
        f"{jac2_url}/user/login/",
        json={"email": email, "password": pw},
    )

    if res.status_code != 200:
        raise RuntimeError(f"Login failed for {email}: {res.text}")

    jac2_session.headers.update({"Content-Type": "application/json"})
    user_token = res.json().get("token", "")
    jac2_session.headers.update({"Authorization": f"bearer {user_token}"})

    jac2_session.post(f"{jac2_url}/walker/init_user", json={})


def load_data(username):
    localpart = username.split("@", 1)[0]
    src_path = Path(f'./raw/{localpart}.json')

    test_data_dir = Path("./test_data")
    qa_dir = Path("./qa")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    test_data_out_path = test_data_dir / f"{localpart}.json"
    qa_out_path = qa_dir / f"{localpart}.json"

    if not src_path.exists():
        print(f"[warn] missing source file: {src_path}")
        return

    with src_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # ---------- auto-detect and extract memory blocks ----------
    if "memories" in data:
        # Old format: direct "memories" list
        memories = data.get("memories", [])
        qa_pairs = data.get("qa_pairs", [])
    elif "reports" in data:
        # Already semi-migrated format
        memories = []
        for r in data["reports"]:
            # either {id, context} or direct dict
            if isinstance(r, dict):
                context = r.get("context", r)
                # ensure memory_id field
                if "memory_id" not in context:
                    context["memory_id"] = r.get("id", f"m{len(memories)+1:02d}")
                memories.append(context)
        qa_pairs = data.get("qa_pairs", [])
    else:
        print(f"[warn] No recognizable memory structure in {src_path.name}")
        return

    # ---------- normalize and rebuild unified structure ----------
    reports = []
    for i, m in enumerate(memories, start=1):
        mem_id = m.get("memory_id") or m.get("id") or f"m{i:02d}"
        summary = m.get("summary", "")
        when = m.get("when", "")
        if isinstance(when, list):
            # take the first element if list provided
            when = when[0] if len(when) > 0 else ""
        elif not isinstance(when, str):
            when = ""
        who = m.get("who", [])
        where = m.get("where", [])
        what = m.get("what", [])
        if isinstance(what, str):
            what = [what]
        session_id = m.get("session_id", "")
        comments_summary = m.get("comments_summary", "")

        context = {
            "memory_id": mem_id,
            "summary": summary,
            "comments_summary": comments_summary,
            "when": when,
            "who": who,
            "where": where,
            "what": what,
            "natural_when": m.get("natural_when", ""),
            "emotion": m.get("emotion", ""),
            "created_at": m.get("created_at", ""),
            "updated_at": m.get("updated_at", ""),
            "image_urls": m.get("image_urls", []),
            "new_image_format": m.get("new_image_format", ""),
            "shared_with": m.get("shared_with", []),
            "conversation": m.get("conversation", []),
            "session_id": session_id,
            "draft": bool(m.get("draft", False)),
        }

        reports.append({
            "id": mem_id,
            "context": context,
        })

    # ---------- build final payload ----------
    migrate_payload = {"status": 200, "reports": reports}
    qa_payload = {"user_id": data.get("user_id", localpart), "qa_pairs": qa_pairs}

    # ---------- save locally ----------
    with test_data_out_path.open("w", encoding="utf-8") as f:
        json.dump(migrate_payload, f, indent=2, ensure_ascii=False)
    with qa_out_path.open("w", encoding="utf-8") as f:
        json.dump(qa_payload, f, indent=2, ensure_ascii=False)

    print(f"[debug] {username}: wrote {len(reports)} memories → {test_data_out_path}")

    # ---------- migrate to backend ----------
    res = jac2_session.post(
        f"{jac2_url}/walker/migrate_profile_data",
        json={"json_file_content": reports},  # pass only the list of reports[].context
    )
    try:
        print(res.json())
    except Exception:
        print({"status_code": res.status_code, "text": res.text})

    if res.status_code == 200:
        print(f"Profile migrated for {username}")

    # ---------- verify ----------
    res = jac2_session.post(f"{jac2_url}/walker/list_memories", json={})
    try:
        print(res.json())
    except Exception:
        print({"status_code": res.status_code, "text": res.text})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("mode", nargs="?", default="run", choices=["migrate", "run"])
    parser.add_argument("--exp", default="mtp", choices=["mtp", "mtp_with_sem", "pe"])
    args = parser.parse_args()

    MODE, EXP = args.mode, args.exp
    users = {f"user_{i:03d}@jaseci.org": "" for i in range(1, 61)}

    for u, pw in users.items():
        users[u] = manage_user_credentials(u, pw)

    if MODE == "migrate":
        for email in sorted(users.keys()):
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

    for run_idx in range(1, 11):  # 10 runs
        run_dir = exp_results_dir / f"run{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)

        all_out_path = run_dir / "all_users_answers.jsonl"
        print(f"\n[RUN {run_idx}] Starting {EXP.upper()} → saving to {run_dir}")

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
                results_path = run_dir / f"{localpart}_answers.jsonl"

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

                        payload = {"query": question}
                        if EXP == "pe":
                            payload["use_byllm"] = False

                        resp = jac2_session.post(
                            f"{jac2_url}/walker/search_memories",
                            json=payload,
                        )
                        
                        try:
                            resp_json = resp.json()
                        except Exception:
                            resp_json = {"status_code": resp.status_code, "text": resp.text}

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

        print(f"[RUN {run_idx}] done. Aggregate → {all_out_path}")

    print(f"\n[all done] Completed 10 runs for experiment: {EXP}")
