# Graph_RAG
  
<b>Description</b>  
This project is a Flask app running GraphRAG for healthcare, made with Vertex AI and Neo4j, to be deployed in a container (Cloud Run or ECS). Initially, a PDF with diseases descriptions is used to enrich the LLM response via RAG. Then, another LLM automatically parses a CSV file with diseases data, generates the Knowledge Graph. After that, an LLM generates a cypher to query the Neo4j KG database and retrieve the possibles diseases, given the patient medical report.  
  
<b>Deployment in Google Cloud Run</b>  
  
Remove user <b>input</b> from `app.py` and get JSON via Flask. 
Add your secrets to Secret Manager
Adapt configurations in` config.json`
    
```
export GCP_PROJECT='your-project'
export GCP_REGION='us-central1'
export AR_REPO='repo-graphrag'
export SERVICE_NAME='flask-app-graphrag'
  
gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"

gcloud run deploy "$SERVICE_NAME" \
     --port=8080 \
     --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
     --allow-unauthenticated \
     --platform=managed  \
     --region=$GCP_REGION \
     --project=$GCP_PROJECT \
     --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION \
     --min-instances 1 --max-instances 5 --cpu 1 --memory 2048Mi --concurrency 10
```
