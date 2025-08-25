snap install google-cloud-cli --classic
source .env
gcloud auth application-default login
gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
gcloud services enable artifactregistry.googleapis.com --project $GOOGLE_CLOUD_PROJECT
gcloud auth configure-docker ${GOOGLE_CLOUD_LOCATION}-docker.pkg.dev
docker build -t europe-west4-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/opennebula-rag-service/opennebula-rag-service:latest .

docker push europe-west4-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/opennebula-rag-service/opennebula-rag-service:latest

curl -X POST "http://localhost:8080/query"      -H "Content-Type: application/json"      -d '{"query": "How do I deploy Kubernetes on OpenNebula?"}'


gcloud run services logs opennebula-rag-app --region europe-west4 --project ${GOOGLE_CLOUD_PROJECT} --follow


curl -X POST "https://opennebula-rag-service-240342877018.europe-west1.run.app/query"      -H "Content-Type: application/json"      -d '{"query": "How do I deploy Kubernetes on OpenNebula?"}'

gcloud run services logs read  opennebula-rag-service --region europe-west1 --project gold-cocoa-469117-a7 
