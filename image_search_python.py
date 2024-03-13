from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
import sys
from io import BytesIO
import json
import os
# from IPython.display import Image
from PIL import Image

import requests


load_dotenv(override=True) # take environment variables from .env.

# Variables not used here do not need to be updated in your .env file
endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else DefaultAzureCredential()
index_name = os.environ["AZURE_SEARCH_INDEX"]
blob_connection_string = os.environ["BLOB_CONNECTION_STRING"]
blob_container_name = os.environ["BLOB_CONTAINER_NAME"]

## List files inside blob container
blob_svc_client = BlobServiceClient.from_connection_string(conn_str=blob_connection_string)
container_client = blob_svc_client.get_container_client(blob_container_name)
blob_list = container_client.list_blobs()

def get_image_vector(image: str, is_local = False) -> list:
    # Define the URL, headers, and data
    url = "https://priya-azure-ai.cognitiveservices.azure.com//computervision/retrieval:vectorizeImage?api-version=2023-02-01-preview&modelVersion=latest"
    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_KEY")
    }

    if not is_local:
        blob_client = blob_svc_client.get_blob_client(container=blob_container_name, blob = image)
        image_data = blob_client.download_blob().readall()

    else:
        print("Running user query")
        with open(image, 'rb') as image_file:
            # Read the contents of the image file
            image_data = image_file.read()

    print(f"Getting vector for {image}...")

    # Send a POST request
    response = requests.post(url, headers=headers, data=image_data)

    # return the vector
    return response.json().get('vector')

descriptions = []
for i, blob in enumerate(blob_list):
   blob_name = blob["name"]
   blob_client = blob_svc_client.get_blob_client(container=blob_container_name, blob = blob_name)
   image_data = blob_client.download_blob().readall()
   image_vector = get_image_vector(blob_name)

   descriptions.append({
       "id" : blob_name.split(".")[0],
       "filename" : blob_name,
       "image_vector" : image_vector,
       "image_path" : f"https://imagesopensourcepriya.blob.core.windows.net/{blob_container_name}/{blob_name}"
   })

with open("image.json", "w") as f:
    json.dump(descriptions, f)
   
## Push local data to the index
from azure.search.documents import SearchClient
import json

# apples_image_directory = os.path.join('..', '..', 'data', 'images', 'apples')
# output_json_file = os.path.join(apples_image_directory, 'output.jsonl')

# data = []
# with open(output_json_file, 'r') as file:
#     for line in file:
#         # Remove leading/trailing whitespace and parse JSON
#         json_data = json.loads(line.strip())
#         data.append(json_data)

## Upload the image embeddings to AI Search
# search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
# results = search_client.upload_documents(descriptions)
# for result in results:
#     print(f'Indexed {result.key} with status code {result.status_code}')

## Search for images
image_filename = "Meesho_sample_query_blue_tshirt.jpg"
# image_path = f"C://Users/priyakedia/OneDrive - Microsoft/Desktop/{image_filename}"

# image = Image.open(image_path)
# image.show()

# image_vector = get_image_vector(r"C://Users/priyakedia/OneDrive - Microsoft/Desktop/Celebs Exposure Shereen Bhan_003.jpg")

def image_search(image_file: str, search_client : SearchClient):

    vector_query = VectorizedQuery(vector=get_image_vector(image_file, is_local=True), k_nearest_neighbors=2, fields="image_vector", exhaustive=True)

    results = search_client.search(
        vector_queries=[vector_query],
        select=["filename", "image_path"],
    )

    for result in results:
        print(result['filename'], result["image_path"], sep=": ")
        response = requests.get(result["image_path"])
        image = Image.open(BytesIO(response.content))
        image.show()

    return results

        # show the image
        # Image(f"./images/{result['fileName']}")
 
# get vector of another image and find closest match
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

## Search the matching images in Azure AI Search
image_filename = "Meesho_sample_query_blue_tshirt.jpg"
search_results = image_search(f"C://Users/priyakedia/OneDrive - Microsoft/Desktop/{image_filename}", search_client)
image_search('pexels-photo-1963641.jpeg')