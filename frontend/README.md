# RAG Legal Frontend v2.4.0

Frontend React para el sistema de consultas legales con IA.

## Características

- Interfaz de chat moderna
- Selector de país (El Salvador, Guatemala, Costa Rica, Panamá, México)
- Sistema de feedback (👍/👎)
- Indicadores de reranking IA y complejidad de query
- Historial de conversaciones
- Estadísticas del sistema
- Diseño responsive

## Desarrollo Local

```bash
# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev

# Build para producción
npm run build
```

## Despliegue en Azure

### Opción 1: Azure Static Web Apps (Recomendado)

```bash
# Instalar Azure SWA CLI
npm install -g @azure/static-web-apps-cli

# Build
npm run build

# Deploy
swa deploy ./dist --env production
```

### Opción 2: Azure Container Apps

```bash
# Build imagen Docker
docker build -t rag-legal-frontend .

# Tag para ACR
docker tag rag-legal-frontend ca5ac7545b05acr.azurecr.io/rag-legal-frontend:latest

# Push a ACR
az acr login --name ca5ac7545b05acr
docker push ca5ac7545b05acr.azurecr.io/rag-legal-frontend:latest

# Crear Container App
az containerapp create \
  --name rag-legal-frontend \
  --resource-group api-graphrag-v21_resource \
  --image ca5ac7545b05acr.azurecr.io/rag-legal-frontend:latest \
  --target-port 80 \
  --ingress external
```

### Opción 3: Azure Blob Storage + CDN

```bash
# Crear storage account
az storage account create \
  --name raglegalfrontend \
  --resource-group api-graphrag-v21_resource \
  --location eastus \
  --sku Standard_LRS

# Habilitar static website
az storage blob service-properties update \
  --account-name raglegalfrontend \
  --static-website \
  --index-document index.html \
  --404-document index.html

# Upload files
az storage blob upload-batch \
  --account-name raglegalfrontend \
  --source ./dist \
  --destination '$web'
```

## Configuración

La URL del API se configura en `src/App.jsx`:

```javascript
const API_URL = 'https://api-graphrag-v21.bravecliff-83b394ec.eastus.azurecontainerapps.io'
```

## Tecnologías

- React 18
- Vite 4
- Tailwind CSS 3
- Lucide React (iconos)
