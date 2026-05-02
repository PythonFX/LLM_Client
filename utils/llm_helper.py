import httpx
import os
from dotenv import load_dotenv
load_dotenv()


def get_azure_ad_token() -> str:
    staff_id = os.environ.get('STAFF_ID')
    staff_pw = os.environ.get('STAFF_PW')
    proxy_url = os.environ.get('AZURE_PROXY_URL')

    proxy = f"http://{staff_id}:{staff_pw}@{proxy_url}" if proxy_url else None

    token_url = os.environ.get("AZURE_TOKEN_URL")
    client_id = os.environ.get('AZURE_CLIENT_ID')
    client_secret = os.environ.get('AZURE_CLIENT_SECRET')
    scope = "https://cognitiveservices.azure.com/.default"

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope
    }

    try:
        with httpx.Client(proxy=proxy, verify=False) as client:
            response = client.post(token_url, data=data)

        if response.status_code == 200:
            token_info = response.json()
            return token_info.get("access_token")
        else:
            print("Failed to get token:", response.status_code)
            print("Response:", response.text)
            return ""
    except Exception as e:
        print(str(e))
        return ""
    
