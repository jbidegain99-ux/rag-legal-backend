#!/usr/bin/env python3
"""
RAG LEGAL LATINOAMÉRICA - Backend API v2.0
Soporte multi-país: MX, SV, GT, CR, PA
"""
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.0.0"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Configuración de países
PAISES = {
    "MX": {
        "nombre": "México",
        "bandera": "🇲🇽",
        "coleccion": "legal_articles_mx",
        "activo": True,
    },
    "SV": {
        "nombre": "El Salvador", 
        "bandera": "🇸🇻",
        "coleccion": "legal_articles_v20",
        "activo": True,
    },
    "GT": {
        "nombre": "Guatemala",
        "bandera": "🇬🇹", 
        "coleccion": "legal_articles_gt",
        "activo": True,
    },
    "CR": {
        "nombre": "Costa Rica",
        "bandera": "🇨🇷",
        "coleccion": "legal_articles_cr",
        "activo": True,
    },
    "PA": {
        "nombre": "Panamá",
        "bandera": "🇵🇦",
        "coleccion": "legal_articles_pa",
        "activo": True,
        "limitado": True,
    },
}

# Límites por plan
PLANES = {
    "free": {"consultas_mes": 10, "paises": ["SV"], "max_tokens": 500},
    "basic": {"consultas_mes": 100, "paises": ["SV", "GT", "CR", "PA"], "max_tokens": 1000},
    "pro": {"consultas_mes": 500, "paises": ["MX", "SV", "GT", "CR", "PA"], "max_tokens": 2000},
    "enterprise": {"consultas_mes": -1, "paises": ["MX", "SV", "GT", "CR", "PA"], "max_tokens": 4000},
}

# ═══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="RAG Legal Latinoamérica",
    description="API de consultas legales con IA - MX, SV, GT, CR, PA",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading
_model = None
_qdrant = None
_openai = None

def get_model():
    global _model
    if _model is None:
        print("📦 Cargando modelo de embeddings...")
        _model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
    return _model

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, timeout=60)
    return _qdrant

def get_openai():
    global _openai
    if _openai is None and OPENAI_KEY:
        _openai = OpenAI(api_key=OPENAI_KEY)
    return _openai

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

class ConsultaRequest(BaseModel):
    query: str
    pais: str = "SV"
    top_k: int = 5
    generar_respuesta: bool = True

class ConsultaResponse(BaseModel):
    query: str
    pais: str
    pais_nombre: str
    articulos: List[Dict[str, Any]]
    respuesta: Optional[str] = None
    tiempo_ms: float

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ═══════════════════════════════════════════════════════════════════════════════

def verificar_acceso(pais: str, plan: str) -> Dict[str, Any]:
    """TEMPORALMENTE DESBLOQUEADO PARA DEMO"""
    return {"permitido": True}
    """Verifica si el plan tiene acceso al país"""
    plan_config = PLANES.get(plan, PLANES["free"])
    if pais not in plan_config["paises"]:
        return {
            "permitido": False,
            "mensaje": f"Tu plan '{plan}' no incluye acceso a {PAISES[pais]['nombre']}. Upgrade a Pro para acceder.",
            "upgrade": "pro" if plan in ["free", "basic"] else None
        }
    return {"permitido": True}

def buscar_articulos(query: str, pais: str, top_k: int = 5) -> List[Dict]:
    """Busca artículos en Qdrant"""
    coleccion = PAISES[pais]["coleccion"]
    model = get_model()
    client = get_qdrant()
    
    embedding = model.encode(query)
    results = client.search(
        collection_name=coleccion,
        query_vector=embedding.tolist(),
        limit=top_k,
        with_payload=True
    )
    
    return [{
        "id": r.payload.get("id", ""),
        "numero": r.payload.get("numero", ""),
        "contenido": r.payload.get("contenido", "")[:1500],
        "codigo": r.payload.get("codigo", ""),
        "score": round(r.score, 4),
        "pais": pais
    } for r in results]

def generar_respuesta(query: str, articulos: List[Dict], pais: str, max_tokens: int = 1000) -> str:
    """Genera respuesta con GPT"""
    openai_client = get_openai()
    if not openai_client or not articulos:
        return ""
    
    pais_nombre = PAISES[pais]["nombre"]
    contexto = "\n\n".join([
        f"**{a['codigo']} - {a['numero']}**\n{a['contenido']}"
        for a in articulos[:5]
    ])
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Eres un experto en derecho de {pais_nombre}. Responde citando artículos específicos."},
                {"role": "user", "content": f"ARTÍCULOS:\n{contexto}\n\nCONSULTA: {query}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error OpenAI: {e}")
        return ""

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "servicio": "RAG Legal Latinoamérica",
        "version": VERSION,
        "paises": [f"{p['bandera']} {p['nombre']}" for p in PAISES.values() if p['activo']],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": VERSION, "timestamp": datetime.now().isoformat()}

@app.get("/api/paises")
async def listar_paises():
    """Lista países con estadísticas"""
    client = get_qdrant()
    resultado = []
    
    for codigo, info in PAISES.items():
        if not info["activo"]:
            continue
        try:
            vectores = client.get_collection(info["coleccion"]).points_count
        except:
            vectores = 0
        
        resultado.append({
            "codigo": codigo,
            "nombre": info["nombre"],
            "bandera": info["bandera"],
            "vectores": vectores,
            "limitado": info.get("limitado", False),
            "planes_acceso": [p for p, c in PLANES.items() if codigo in c["paises"]]
        })
    
    return {"paises": resultado}

@app.get("/api/planes")
async def listar_planes():
    """Lista planes disponibles"""
    precios = {"free": 0, "basic": 9.99, "pro": 29.99, "enterprise": 99.99}
    return {
        "planes": [{
            "id": pid,
            "consultas_mes": cfg["consultas_mes"],
            "paises": cfg["paises"],
            "paises_nombres": [PAISES[p]["nombre"] for p in cfg["paises"]],
            "precio_usd": precios.get(pid, 0)
        } for pid, cfg in PLANES.items()]
    }

@app.get("/api/stats")
async def estadisticas():
    """Estadísticas del sistema"""
    client = get_qdrant()
    stats = {}
    total = 0
    
    for codigo, info in PAISES.items():
        try:
            vectores = client.get_collection(info["coleccion"]).points_count
            stats[codigo] = {"nombre": info["nombre"], "bandera": info["bandera"], "vectores": vectores}
            total += vectores
        except:
            pass
    
    return {"version": VERSION, "paises_activos": len(stats), "total_vectores": total, "por_pais": stats}

@app.post("/api/consulta")
async def consulta(
    request: ConsultaRequest,
    x_user_id: Optional[str] = Header(None),
    x_user_plan: Optional[str] = Header(None, alias="x-user-plan")
):
    """Consulta legal principal"""
    inicio = time.time()
    pais = request.pais.upper()
    
    # Validar país
    if pais not in PAISES or not PAISES[pais]["activo"]:
        raise HTTPException(status_code=400, detail=f"País no soportado: {pais}")
    
    # Verificar acceso por plan
    plan = x_user_plan or "free"
    acceso = verificar_acceso(pais, plan)
    if not acceso["permitido"]:
        raise HTTPException(status_code=403, detail={
            "error": "pais_no_disponible",
            "mensaje": acceso["mensaje"],
            "upgrade_sugerido": acceso.get("upgrade")
        })
    
    # Buscar artículos
    articulos = buscar_articulos(request.query, pais, request.top_k)
    
    # Generar respuesta
    respuesta = None
    if request.generar_respuesta and articulos:
        max_tokens = PLANES.get(plan, PLANES["free"])["max_tokens"]
        respuesta = generar_respuesta(request.query, articulos, pais, max_tokens)
    
    return ConsultaResponse(
        query=request.query,
        pais=pais,
        pais_nombre=PAISES[pais]["nombre"],
        articulos=articulos,
        respuesta=respuesta,
        tiempo_ms=round((time.time() - inicio) * 1000, 2)
    )

# Endpoint legacy para compatibilidad
@app.post("/api/query")
async def query_legacy(request: ConsultaRequest):
    """Endpoint legacy - redirige a /api/consulta"""
    return await consulta(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS DE COMPATIBILIDAD - CONVERSACIONES
# ═══════════════════════════════════════════════════════════════════════════════

# Almacenamiento temporal en memoria (para producción usar base de datos)
_conversations = {}

@app.get("/api/conversations")
async def listar_conversaciones(x_user_id: Optional[str] = Header(None)):
    """Lista conversaciones del usuario"""
    user_id = x_user_id or "anonymous"
    user_convs = [c for c in _conversations.values() if c.get("user_id") == user_id]
    return {"conversations": user_convs}

@app.post("/api/conversations")
async def crear_conversacion(x_user_id: Optional[str] = Header(None)):
    """Crea nueva conversación"""
    import uuid
    conv_id = str(uuid.uuid4())
    user_id = x_user_id or "anonymous"
    
    conv = {
        "id": conv_id,
        "user_id": user_id,
        "title": "Nueva conversación",
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    _conversations[conv_id] = conv
    return conv

@app.get("/api/conversations/{conv_id}")
async def obtener_conversacion(conv_id: str):
    """Obtiene una conversación por ID"""
    if conv_id not in _conversations:
        # Crear conversación si no existe
        _conversations[conv_id] = {
            "id": conv_id,
            "user_id": "anonymous",
            "title": "Conversación",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    return _conversations[conv_id]

@app.put("/api/conversations/{conv_id}")
async def actualizar_conversacion(conv_id: str, data: dict):
    """Actualiza una conversación"""
    if conv_id in _conversations:
        _conversations[conv_id].update(data)
        _conversations[conv_id]["updated_at"] = datetime.now().isoformat()
    return _conversations.get(conv_id, {"id": conv_id})

@app.delete("/api/conversations/{conv_id}")
async def eliminar_conversacion(conv_id: str):
    """Elimina una conversación"""
    if conv_id in _conversations:
        del _conversations[conv_id]
    return {"deleted": True}

@app.post("/api/conversations/{conv_id}/messages")
async def agregar_mensaje(conv_id: str, data: dict):
    """Agrega mensaje a conversación"""
    import uuid
    if conv_id not in _conversations:
        _conversations[conv_id] = {
            "id": conv_id,
            "user_id": "anonymous", 
            "title": "Conversación",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    message = {
        "id": str(uuid.uuid4()),
        "content": data.get("content", ""),
        "role": data.get("role", "user"),
        "created_at": datetime.now().isoformat()
    }
    _conversations[conv_id]["messages"].append(message)
    _conversations[conv_id]["updated_at"] = datetime.now().isoformat()
    
    return message

# Endpoint legacy /api/config para compatibilidad
@app.get("/api/config")
async def config():
    """Configuración del sistema (compatibilidad)"""
    client = get_qdrant()
    total = 0
    for info in PAISES.values():
        try:
            total += client.get_collection(info["coleccion"]).points_count
        except:
            pass
    
    return {
        "version": VERSION,
        "total_articulos": total,
        "num_conceptos": 0,
        "embedding_model": "hiiamsid/sentence_similarity_spanish_es",
        "paises_activos": len(PAISES)
    }
