# google_auth.py
import os
import json
import google.auth
import google.auth.transport.requests
import google_auth_oauthlib.flow
import googleapiclient.discovery
from google.auth.exceptions import DefaultCredentialsError

# Chemin vers votre fichier credentials.json
CLIENT_SECRET_FILE = 'CODE/credentials.json'

# Scopes dont nous avons besoin
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Nom du fichier token.json pour stocker les jetons d'accès et de rafraîchissement
TOKEN_FILE = 'token.json'

def authenticate():
    creds = None
    # Vérifiez si le fichier token.json existe
    if os.path.exists(TOKEN_FILE):
        try:
            creds = google.auth.load_credentials_from_file(TOKEN_FILE, scopes=SCOPES)[0]
        except DefaultCredentialsError:
            os.remove(TOKEN_FILE)
            creds = None
    # Si il n'existe pas ou s'il n'est pas valide, authentifiez l'utilisateur et obtenez de nouveaux jetons
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Sauvegardez les identifiants pour la prochaine exécution
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

def create_drive_service():
    creds = authenticate()
    service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)
    return service

if __name__ == "__main__":
    service = create_drive_service()
    print("Google Drive service created successfully")
