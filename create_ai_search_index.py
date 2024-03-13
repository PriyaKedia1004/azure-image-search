from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex
)

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential, DefaultAzureCredential
import os

endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else DefaultAzureCredential()
index_name = os.environ["AZURE_SEARCH_INDEX"]

# Create a search index 
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)  
fields = [  
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),  
    SearchField(name="filename", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),  
    SearchField(
        name="image_vector",  
        hidden=True,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
        searchable=True,
        vector_search_dimensions=1024,  
        vector_search_profile_name="myHnswProfile"
    ), 
    SearchField(name="image_path", type=SearchFieldDataType.String, searchable=True, sortable=False, filterable=True, facetable=True), 
]  
  
# Configure the vector search configuration  
vector_search = VectorSearch(  
    algorithms=[  
        HnswAlgorithmConfiguration(  
            name="myHnsw"
        )
    ],  
   profiles=[  
        VectorSearchProfile(  
            name="myHnswProfile",  
            algorithm_configuration_name="myHnsw",  
        )
    ],  
)  
  
# Create the search index with the vector search configuration  
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)  
result = index_client.create_or_update_index(index)  
print(f"{result.name} created") 
