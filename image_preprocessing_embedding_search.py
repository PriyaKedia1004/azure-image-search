from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from msrest.authentication import CognitiveServicesCredentials
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
import os
import json
from PIL import Image
from io import BytesIO
import requests

from dotenv import load_dotenv
import os

load_dotenv()

## Load the environment variables
azure_ai_search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
azure_ai_search_credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])
index_name = os.environ["AZURE_SEARCH_INDEX"]
vision_endpoint = os.getenv("VISION_ENDPOINT")
vision_key = os.getenv("VISION_KEY")

def get_roi_stream(image_path):
    vision_client = ImageAnalysisClient(endpoint=vision_endpoint, credential=AzureKeyCredential(vision_key))
    visual_features = [VisualFeatures.PEOPLE]

    with open(image_path, "rb") as f:
        image_data = f.read()

    result = vision_client.analyze(image_data=image_data, visual_features=visual_features)

    if result.people is not None:
        print(" People:")
        person = result.people.list[0]
        x, y, w, h = person.bounding_box.values()
        image = Image.open(image_path)
        ## Define the bounding boxes
        bbox = (x, y, x+w, y+h)
        ## Crop the image to the bounding box
        roi_image = image.crop(bbox)
        roi_image_stream = BytesIO()
        roi_image.save(roi_image_stream, format = 'JPEG')
        roi_image_stream = roi_image_stream.getvalue()

    return roi_image_stream

def get_vector_from_image_stream(image_stream) -> list:
    # Define the URL, headers, and data
    url = f"{azure_ai_endpoint}/computervision/retrieval:vectorizeImage?api-version=2023-02-01-preview&modelVersion=latest"
    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_KEY")
    }

    # Send a POST request
    response = requests.post(url, headers=headers, data=image_stream)

    # return the vector
    return response.json().get('vector')

def image_search(image_stream, search_client : SearchClient):

    vector_query = VectorizedQuery(vector=get_vector_from_image_stream(image_stream), k_nearest_neighbors=5, fields="image_vector", exhaustive=True)

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

def remove_background_from_image(image_stream):
    # Update the endpoint uri
    endpoint = f"{azure_ai_endpoint}/computervision/imageanalysis:segment?api-version=2023-02-01-preview"
    background_removal = "&mode=backgroundRemoval"

    headers = {
        'Content-type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': os.getenv("AZURE_AI_KEY")
    }

    r = requests.post(endpoint + background_removal, data = image_stream, headers=headers)

    print("Background from image removed completely")

    return r.content

## Get closest match from the marketplace
search_client = SearchClient(endpoint=azure_ai_search_endpoint, index_name=index_name, credential=azure_ai_search_credential)
roi_image_stream = get_roi_stream(image_path)
roi_image_stream = remove_background_from_image(roi_image_stream)
image_search(roi_image_stream, search_client)
