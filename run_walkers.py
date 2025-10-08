import json
import os
import requests
import secrets
import time


jac2_url = "http://localhost:8000"

def list_user_memories(users: list[str] = [],email=""):
    if len(users) == 0:
        with open("data/user_masters.json", "r") as f:
            users = json.load(f)
        users = [u["user"] for u in users]

    with open("data/new_user_creds.json", "r") as f:
        new_user_creds = json.load(f)

    for user in users:
        jac2_session = requests.Session()
        res = jac2_session.post(
            f"{jac2_url}/user/login/",
            json={"email": user, "password": new_user_creds[user]},
        )

        jac2_session.headers.update({"Content-Type": "application/json"})

        jac2_session.headers.update(
            {"Authorization": f"bearer {res.json()['token']}"}
        )

        res = jac2_session.post(f"{jac2_url}/walker/list_memories", json={})
        print(user,":\n",res.text,"\n")

def retrive_memories(users: list[str] = [],email=""):
    if len(users) == 0:
        with open("data/user_masters.json", "r") as f:
            users = json.load(f)
        users = [u["user"] for u in users]

    with open("data/new_user_creds.json", "r") as f:
        new_user_creds = json.load(f)

    for user in users:
        if user == email :
            jac2_session = requests.Session()
            res = jac2_session.post(
                f"{jac2_url}/user/login/",
                json={"email": user, "password": new_user_creds[user]},
            )

            jac2_session.headers.update({"Content-Type": "application/json"})

            jac2_session.headers.update(
                {"Authorization": f"bearer {res.json()['token']}"}
            )

            with open(f"data/{user}/conversation.json", "r") as f:
                utterance = json.load(f)

            #utterance="I want a memory on spending time on a beach?"

            res = jac2_session.post(f"{jac2_url}/walker/memory_retrieval/", json={'utterance':utterance})
            print(user,":\n",res,"\n")

def identify_common_interets(users: list[str] = [],email=""):
    if len(users) == 0:
        with open("data/user_masters.json", "r") as f:
            users = json.load(f)
        users = [u["user"] for u in users]

    with open("data/new_user_creds.json", "r") as f:
        new_user_creds = json.load(f)

    for user in users:
        if user == email :
            jac2_session = requests.Session()
            res = jac2_session.post(
                f"{jac2_url}/user/login/",
                json={"email": user, "password": new_user_creds[user]},
            )

            jac2_session.headers.update({"Content-Type": "application/json"})

            jac2_session.headers.update(
                {"Authorization": f"bearer {res.json()['token']}"}
            )

            #res = jac2_session.post(f"{jac2_url}/walker/analyze_user/", json={})

            with open(f"data/{user}/conversation.json", "r") as f:
                utterance = json.load(f)            

            res = jac2_session.post(f"{jac2_url}/walker/analyze_user/", json={})
            res = jac2_session.post(f"{jac2_url}/walker/chat_about_memories_interests/", json={'utterance':utterance})
            # print(user,":\n",res,"\n")
            #print(res.json())
            relevant_memories = res.json()['reports'][0]

            for i in range (5):
                with open(f"data/{user}/conversation.json", "r") as f:
                    utterance = json.load(f)   
                res = jac2_session.post(f"{jac2_url}/walker/chat_about_memories/", json={'utterance':utterance,'relevant_memories':relevant_memories})
                print( res.json()['reports'][1])
                chat_input = input("Your Response : ")
                chat_history = res.json()['reports'][0]
                chat_history["conversation"].append({"role": "user", "content": chat_input})
                print(chat_history)
                with open(f"data/{user}/conversation.json", "w") as f:
                    json.dump(chat_history, f, indent=4)
                    f.flush()

if __name__ == "__main__":
    #list_user_memories()
    #retrive_memories(email="pat@tobu.life")
    identify_common_interets(email="pat@tobu.life")
