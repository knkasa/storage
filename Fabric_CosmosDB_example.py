#%pip install -q azure-cosmos

#============ Normal DB ========================
COSMOS_ENDPOINT = "https://1b6f18bc-deaf-48ec-8441-435f2fed06f7.z1b.sql.cosmos.fabric.microsoft.com:443/"
DATABASE_NAME   = "cosmos_db_test"
CONTAINER_NAME  = "db1"

from notebookutils import mssparkutils
from azure.core.credentials import AccessToken
from azure.cosmos.aio import CosmosClient # Use async version

scope = "https://cosmos.azure.com/.default"

access_token: str = mssparkutils.credentials.getToken(scope)

class FabricTokenCredential:
    def get_token(self, *scopes, **kwargs):
        return AccessToken(access_token, 0)

credential = FabricTokenCredential()

import asyncio
from azure.cosmos import PartitionKey

async def run():
    async with CosmosClient(COSMOS_ENDPOINT, credential=credential) as client:
        db = await client.create_database_if_not_exists(id=DATABASE_NAME) #client.get_database_client(DATABASE_NAME)
        database = client.get_database_client(DATABASE_NAME)

        await database.create_container_if_not_exists(
            id=CONTAINER_NAME,
            partition_key=PartitionKey(path="/type")
        )
        container = database.get_container_client(CONTAINER_NAME)

        docs = [
            {"id": "p1", "type": "Person",  "name": "Ada Lovelace", "born": 1815},
            {"id": "c1", "type": "Company", "name": "Contoso",      "founded": 1996},
        ]
        for d in docs:
            await container.upsert_item(d)

        res = [i async for i in container.query_items(query="SELECT c.id, c.type, c.name FROM c")]

        return res

items = await run()
print(items)


#============== Vector DB ==========================
COSMOS_ENDPOINT = "https://1b6f18bc-deaf-48ec-8441-435f2fed06f7.z1b.sql.cosmos.fabric.microsoft.com:443/"
DATABASE_NAME   = "cosmos_db_test"
CONTAINER_NAME  = "vector_db1"

from notebookutils import mssparkutils
from azure.core.credentials import AccessToken
from azure.cosmos import CosmosClient  # Use sync version.

scope = "https://cosmos.azure.com/.default"

access_token: str = mssparkutils.credentials.getToken(scope)

class FabricTokenCredential:
    def get_token(self, *scopes, **kwargs):
        return AccessToken(access_token, 0)

credential = FabricTokenCredential()

client    = CosmosClient(COSMOS_ENDPOINT, credential=credential)
database  = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

import random
import uuid

def rand_vec4():
    return [round(random.uniform(-1, 1), 6) for _ in range(4)]

docs = [
    {
        "id": str(uuid.uuid4()),
        "title": "Item A",
        "vector1": rand_vec4(),     # <-- must match your Container Vector Policy path (e.g., /vector1)
        "tags": ["sample","fabric","cosmosdb"]
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Item B",
        "vector1": rand_vec4(),
        "tags": ["sample"]
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Item C",
        "vector1": rand_vec4(),
        "tags": ["demo"]
    },
]

for d in docs:
    container.upsert_item(d)

# Make a random query vector (dimension=4)
query_vec = rand_vec4()

sql = """
SELECT TOP 3
  c.id,
  c.title,
  c.vector1,
  VECTORDISTANCE(c.vector1, @q) AS similarity
FROM c
ORDER BY VECTORDISTANCE(c.vector1, @q)
"""

params = [{"name": "@q", "value": query_vec}]

results = list(container.query_items(
    query=sql,
    parameters=params,
    enable_cross_partition_query=True
))

print("Query vector:", query_vec)
for r in results:
    print(r["title"], r["similarity"], r["vector1"])


#============== Knowlege Graph ==========================
COSMOS_ENDPOINT = "https://1b6f18bc-deaf-48ec-8441-435f2fed06f7.z1b.sql.cosmos.fabric.microsoft.com:443/"
DATABASE_NAME   = "cosmos_db_test"
CONTAINER_NAME  = "kg1"

from notebookutils import mssparkutils
from azure.core.credentials import AccessToken
from azure.cosmos import CosmosClient  # Use sync version.

scope = "https://cosmos.azure.com/.default"

access_token: str = mssparkutils.credentials.getToken(scope)

class FabricTokenCredential:
    def get_token(self, *scopes, **kwargs):
        return AccessToken(access_token, 0)

credential = FabricTokenCredential()

client = CosmosClient(COSMOS_ENDPOINT, credential=credential)      # or CosmosClient(ENDPOINT, DefaultAzureCredential())
db = client.get_database_client(DATABASE_NAME)
container = db.get_container_client(CONTAINER_NAME)

import uuid

def v_id(label, local_id): return f"{label}:{local_id}"
def e_id(src, dst, label): return f"E:{src}->{dst}:{label}"

docs = [
    # vertices
    {"id": v_id("P", "alice"), "type":"vertex", "label":"Person", "pk":"graph1",
     "name":"Alice", "title":"Researcher", "props":{"hobbies":["ml","tennis"]}},
    {"id": v_id("P", "bob"),   "type":"vertex", "label":"Person", "pk":"graph1",
     "name":"Bob",   "title":"PM", "props":{"hobbies":["ml","golf"]}},
    {"id": v_id("D", "paper123"), "type":"vertex", "label":"Doc", "pk":"graph1",
     "title":"Vector Search in Cosmos DB", "year":2025},

    # edges
    {"id": e_id(v_id("P","alice"), v_id("D","paper123"), "AUTHORED"),
     "type":"edge", "label":"AUTHORED", "pk":"graph1",
     "fromId": v_id("P","alice"), "toId": v_id("D","paper123"), "since":"2025-01-01"},

    {"id": e_id(v_id("P","alice"), v_id("P","bob"), "KNOWS"),
     "type":"edge", "label":"KNOWS", "pk":"graph1",
     "fromId": v_id("P","alice"), "toId": v_id("P","bob"), "strength":0.9}
]

for d in docs:
    container.upsert_item(d)

print("Inserted/updated", len(docs), "items")

#Choose the target node(edge). pk=partition_key
person_id = v_id("P", "alice")

sql_edges = """
SELECT e.toId, e.label
FROM e
WHERE e.type = "edge" AND e.pk = @pk AND e.fromId = @fromId
"""

edges = list(container.query_items(
    query=sql_edges,
    parameters=[{"name":"@pk","value":"graph1"}, {"name":"@fromId","value":person_id}],
    partition_key="graph1"             # ✅ route to a single partition
))

to_ids = [e["toId"] for e in edges]

neighbors = []
for vid in to_ids:
    neighbors.append(container.read_item(item=vid, partition_key="graph1"))

print(neighbors)
