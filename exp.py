users = {"yipingtest3@jaseci.org": ""}
name = "yipingtest3@jaseci.org"
jac2_url = "http://localhost:8000"
user_token = ""


import random
import string
import json
import os
import datetime
import requests


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
    load_data(username)
    

def load_data(username):

    with open(f'./yipingtest3@jaseci.org/memory.json', 'r') as f:
        profile_json = json.load(f)

    # print(profile_json["reports"])

    res = jac2_session.post(
        f"{jac2_url}/walker/migrate_profile_data",
        json={"json_file_content": profile_json["reports"]},
    )
    # print(res.json())
    if res.status_code == 200:
        print(f"Profile migrated for {user}")
    # jac2_session.headers.update({"Authorization": f"bearer {user_token}"})
    res = jac2_session.post(f"{jac2_url}/walker/list_memories", json={})
    print(res.json())

# Example usage:
if __name__ == "__main__":
    for user, password in users.items():
        new_password = manage_user_credentials(user, password)
        users[user] = new_password  # Update the dictionary with the password

    print (users)
    login_user(name, users[name])

    res = jac2_session.post(f"{jac2_url}/walker/list_memories", json={})
    # print(res.json())