# drive_upload.py
from google_auth import create_drive_service
import googleapiclient.http
import os

def upload_file(service, file_path, file_name, mime_type='image/png'):
    file_metadata = {'name': file_name}
    media = googleapiclient.http.MediaFileUpload(file_path, mimetype=mime_type)
    
    # Créer un fichier sur Google Drive
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File ID: {file.get("id")}')
    return file.get('id')

def create_shareable_link(service, file_id):
    # Changer les permissions pour rendre le fichier public
    def set_permission(file_id):
        permissions = {
            'role': 'reader',
            'type': 'anyone'
        }
        service.permissions().create(fileId=file_id, body=permissions).execute()
    
    set_permission(file_id)
    
    # Obtenir le lien partageable
    link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return link

def upload_images_in_directory(service, directory_path):
    links = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith('.png'):
            print(f'Uploading file: {file_name}')
            file_id = upload_file(service, file_path, file_name)
            shareable_link = create_shareable_link(service, file_id)
            links.append((file_name, shareable_link))
    return links

if __name__ == "__main__":
    # Chemin vers votre dossier contenant les images
    directory_path = rf'C:\Users\PAYA Elsa\Documents\ELSA\ECOLE\S8 M1\MASTER PROJECT\CODE\backend\CODE\ML_Models\random_forest_csv\Prev_Images'

    # Créer un service Google Drive
    service = create_drive_service()

    # Uploader les images et obtenir les liens partageables
    links = upload_images_in_directory(service, directory_path)
    
    # Afficher les liens partageables
    for file_name, link in links:
        print(f'File: {file_name} - Shareable Link: {link}')
