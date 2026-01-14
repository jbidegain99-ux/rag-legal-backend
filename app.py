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
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from openai import OpenAI
import re

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.3.0"  # Fix: HyDE templates mejorados + Endpoint de feedback
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
# DETECCIÓN INTELIGENTE DE TIPO DE CÓDIGO LEGAL
# ═══════════════════════════════════════════════════════════════════════════════

# Términos que indican claramente un tipo de código específico
TERMINOS_CODIGO_PENAL = {
    # Delitos contra la vida
    "homicidio", "asesinato", "parricidio", "femicidio", "feminicidio", "infanticidio",
    # Delitos contra la integridad
    "lesiones", "agresión", "violencia", "golpes", "maltrato",
    # Delitos contra la propiedad
    "robo", "hurto", "estafa", "fraude", "extorsión", "chantaje", "receptación",
    "apropiación indebida", "usurpación", "daños", "incendio",
    # Delitos sexuales
    "violación", "abuso sexual", "acoso sexual", "estupro",
    # Delitos contra la libertad
    "secuestro", "privación de libertad", "trata de personas", "amenazas", "coacciones",
    # Delitos contra la seguridad
    "terrorismo", "narcotráfico", "tráfico de drogas", "portación de armas",
    # Causas de justificación y eximentes
    "legítima defensa", "legitima defensa", "defensa propia", "estado de necesidad",
    "inimputabilidad", "eximente", "atenuante", "agravante",
    # Penas y medidas
    "prisión", "cárcel", "pena de muerte", "cadena perpetua", "multa penal",
    "libertad condicional", "libertad vigilada",
    # Procedimiento penal
    "denuncia penal", "querella", "acusación", "imputado", "procesado",
    "sentencia penal", "condena", "absolución",
    # Términos generales penales
    "delito", "crimen", "criminal", "delincuente", "pena", "sanción penal",
    "código penal", "penal", "tipicidad", "antijuridicidad", "culpabilidad",
    "dolo", "culpa", "tentativa", "consumación", "autoría", "participación",
    "cómplice", "encubrimiento"
}

TERMINOS_CODIGO_CIVIL = {
    # Personas
    "capacidad civil", "incapacidad", "menor de edad", "emancipación",
    "persona jurídica", "persona natural",
    # Familia
    "matrimonio", "divorcio", "separación", "nulidad matrimonial",
    "patria potestad", "custodia", "alimentos", "pensión alimenticia",
    "adopción", "filiación", "paternidad", "maternidad",
    # Obligaciones y contratos
    "contrato", "obligación civil", "compraventa", "arrendamiento",
    "préstamo", "hipoteca", "fianza", "depósito", "mandato",
    "incumplimiento contractual", "resolución de contrato",
    # Bienes y propiedad
    "propiedad", "posesión", "usufructo", "servidumbre",
    "bienes muebles", "bienes inmuebles", "registro de propiedad",
    # Sucesiones
    "herencia", "testamento", "sucesión", "legado", "heredero",
    "albacea", "partición hereditaria",
    # Responsabilidad civil
    "responsabilidad civil", "daños y perjuicios", "indemnización civil",
    # Términos generales civiles
    "código civil", "civil", "derecho civil"
}

TERMINOS_CODIGO_LABORAL = {
    "despido", "contrato de trabajo", "salario", "sueldo", "vacaciones",
    "aguinaldo", "indemnización laboral", "jornada laboral", "horas extras",
    "sindicato", "huelga", "patrono", "empleador", "trabajador", "empleado",
    "seguridad social", "pensión", "jubilación", "accidente laboral",
    "enfermedad profesional", "código de trabajo", "laboral", "trabajo"
}

TERMINOS_CODIGO_PROCESAL = {
    "demanda", "contestación", "prueba", "sentencia", "apelación",
    "recurso", "casación", "amparo", "habeas corpus", "medida cautelar",
    "embargo", "notificación", "citación", "audiencia", "juicio",
    "proceso", "procedimiento", "jurisdicción", "competencia",
    "código procesal", "procesal"
}

TERMINOS_CONSTITUCION = {
    "constitución", "constitucional", "derechos fundamentales",
    "garantías constitucionales", "inconstitucionalidad", "amparo constitucional"
}

# Mapeo de códigos detectados a valores del campo "codigo" en Qdrant
# IMPORTANTE: Estos valores deben coincidir EXACTAMENTE con los de la base de datos
CODIGO_MAPPING = {
    "PENAL": ["Codigo Penal", "Codigo Procesal Penal"],
    "CIVIL": ["Codigo Civil", "Codigo De Familia", "Codigo Procesal Civil Y Mercantil"],
    "LABORAL": ["Codigo De Trabajo"],
    "PROCESAL": ["Codigo Procesal Penal", "Codigo Procesal Civil Y Mercantil"],
    "CONSTITUCION": ["Constitucion"],
    "COMERCIAL": ["Codigo Comercio"],
    "TRIBUTARIO": ["Codigo Tributario"],
}


def detectar_tipo_codigo(query: str) -> Optional[str]:
    """
    Detecta el tipo de código legal basado en términos en la consulta.
    Retorna: 'PENAL', 'CIVIL', 'LABORAL', 'PROCESAL', 'CONSTITUCION' o None
    """
    query_lower = query.lower()

    # Buscar menciones explícitas primero
    if any(term in query_lower for term in ["código penal", "codigo penal", "c. penal", "cp "]):
        return "PENAL"
    if any(term in query_lower for term in ["código civil", "codigo civil", "c. civil", "cc "]):
        return "CIVIL"
    if any(term in query_lower for term in ["código de trabajo", "codigo de trabajo", "laboral"]):
        return "LABORAL"
    if any(term in query_lower for term in ["constitución", "constitucion", "constitucional"]):
        return "CONSTITUCION"

    # Contar coincidencias por categoría
    scores = {
        "PENAL": sum(1 for term in TERMINOS_CODIGO_PENAL if term in query_lower),
        "CIVIL": sum(1 for term in TERMINOS_CODIGO_CIVIL if term in query_lower),
        "LABORAL": sum(1 for term in TERMINOS_CODIGO_LABORAL if term in query_lower),
        "PROCESAL": sum(1 for term in TERMINOS_CODIGO_PROCESAL if term in query_lower),
        "CONSTITUCION": sum(1 for term in TERMINOS_CONSTITUCION if term in query_lower),
    }

    # Retornar el tipo con mayor score si hay al menos una coincidencia
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores, key=scores.get)

    return None


def enriquecer_query(query: str, tipo_codigo: str) -> str:
    """
    Enriquece la query con contexto legal para mejorar la búsqueda semántica.
    """
    contexto = {
        "PENAL": "derecho penal delito sanción pena",
        "CIVIL": "derecho civil obligación contrato",
        "LABORAL": "derecho laboral trabajo empleado",
        "PROCESAL": "procedimiento judicial proceso",
        "CONSTITUCION": "derecho constitucional garantías"
    }

    if tipo_codigo and tipo_codigo in contexto:
        return f"{query} {contexto[tipo_codigo]}"
    return query


def crear_filtro_codigo(tipo_codigo: str) -> Optional[Filter]:
    """
    Crea un filtro de Qdrant para el tipo de código detectado.
    Usa MatchAny para buscar cualquiera de los valores posibles.
    """
    if not tipo_codigo or tipo_codigo not in CODIGO_MAPPING:
        return None

    valores_codigo = CODIGO_MAPPING[tipo_codigo]

    return Filter(
        must=[
            FieldCondition(
                key="codigo",
                match=MatchAny(any=valores_codigo)
            )
        ]
    )

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
    tipo_codigo_detectado: Optional[str] = None  # Tipo de código detectado (PENAL, CIVIL, etc.)

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

def generar_query_hyde(query: str, tipo_codigo: str) -> str:
    """
    Genera una query expandida estilo HyDE (Hypothetical Document Embeddings).
    En lugar de buscar solo "robo", genera un documento hipotético que describe
    cómo se vería un artículo sobre robo.
    """
    templates_hyde = {
        "PENAL": {
            # Delitos contra la propiedad - MUY específicos
            "robo": "Artículo sobre el delito de ROBO. El que con ánimo de lucro se apoderare de cosa mueble ajena empleando FUERZA EN LAS COSAS o VIOLENCIA O INTIMIDACIÓN en las personas, será sancionado con prisión de seis a diez años. Robo agravado cuando se comete con arma, en casa habitada, o por dos o más personas.",
            "hurto": "Artículo sobre el delito de HURTO. El que con ánimo de lucro se apoderare de cosa mueble ajena SIN FUERZA ni violencia será sancionado con prisión de dos a cinco años. El hurto se diferencia del robo porque NO hay fuerza ni intimidación. Hurto simple, hurto agravado.",

            # Delitos contra la vida - Diferenciar tipos de homicidio
            "homicidio": "Artículo sobre el delito de HOMICIDIO SIMPLE. El que matare a otro será sancionado con prisión de diez a veinte años. Homicidio doloso intencional. Art. 128, 129, 130.",
            "homicidio simple": "Artículo sobre HOMICIDIO SIMPLE intencional doloso. El que matare a otro con intención será sancionado con prisión. Art. 128, 129. Homicidio básico sin agravantes.",
            "homicidio culposo": "Artículo sobre HOMICIDIO CULPOSO. El que por culpa causare la muerte de otro será sancionado. Muerte por imprudencia, negligencia o impericia. Art. 132. Sin intención de matar.",
            "homicidio agravado": "Artículo sobre HOMICIDIO AGRAVADO. El homicidio será sancionado con prisión de veinte a treinta años cuando concurrieren circunstancias agravantes. Asesinato, alevosía, ensañamiento. Art. 129.",

            # Delitos contra la libertad - MUY específico
            "secuestro": "Artículo sobre el delito de SECUESTRO. El que privare a otro de su libertad individual con el propósito de obtener rescate, cumplimiento de condición o para otros fines, será sancionado con prisión de diez a veinte años. Privación de libertad con fines específicos. Art. 149, 150, 151. Secuestro extorsivo.",
            "privación de libertad": "Artículo sobre PRIVACIÓN DE LIBERTAD. El que privare a otro de su libertad personal será sancionado con prisión. Detención ilegal, encierro. Delitos contra la libertad individual.",

            # Delitos sexuales
            "violación": "Artículo sobre el delito de VIOLACIÓN. El que mediante violencia tuviere acceso carnal por vía vaginal o anal con otra persona será sancionado con prisión de seis a diez años. Violación sexual, agresión sexual.",

            # Otros delitos
            "estafa": "Artículo sobre el delito de ESTAFA. El que mediante engaño, artificio o ardid obtuviere para sí o para tercero un provecho injusto en perjuicio ajeno será sancionado. Defraudación, fraude. Art. 215, 216.",
            "extorsión": "Artículo sobre el delito de EXTORSIÓN. El que mediante intimidación o amenaza obligare a otro a realizar, tolerar u omitir un acto en perjuicio de su patrimonio será sancionado. Art. 214.",

            # Causas de justificación
            "legítima defensa": "Artículo sobre LEGÍTIMA DEFENSA como causa de justificación y eximente de responsabilidad penal. No es punible quien actúa en defensa de su persona, honor o bienes, o de terceros, repeliendo una agresión ilegítima, actual o inminente. Art. 27. Defensa propia.",

            "default": "Artículo del Código Penal sobre delitos y penas. Sanción penal, prisión, multa, responsabilidad criminal."
        },
        "CIVIL": {
            "matrimonio": "Artículo sobre el MATRIMONIO. El matrimonio es la unión legal de un hombre y una mujer con el fin de establecer una plena comunidad de vida. Requisitos, impedimentos, celebración.",
            "divorcio": "Artículo sobre el DIVORCIO. El divorcio disuelve el vínculo matrimonial. Causales de divorcio, divorcio por mutuo consentimiento, efectos patrimoniales.",
            "patria potestad": "Artículo sobre la PATRIA POTESTAD. La patria potestad es el conjunto de derechos y deberes que tienen los padres sobre los hijos menores. Autoridad parental, guarda, custodia.",
            "herencia": "Artículo sobre la HERENCIA y sucesiones. La herencia comprende todos los bienes, derechos y obligaciones del causante. Sucesión testamentaria, sucesión intestada.",
            "default": "Artículo del Código Civil o de Familia sobre obligaciones, contratos, propiedad, familia, sucesiones."
        },
        "LABORAL": {
            "despido": "Artículo sobre el DESPIDO laboral. El patrono puede dar por terminado el contrato sin responsabilidad por causas justificadas. Despido justificado, injustificado, indemnización.",
            "vacaciones": "Artículo sobre las VACACIONES del trabajador. Todo trabajador tiene derecho a un período de vacaciones remuneradas después de cada año de trabajo continuo. Art. 177. Quince días de vacaciones.",
            "aguinaldo": "Artículo sobre el AGUINALDO. Todo patrono está obligado a dar a sus trabajadores una prima por cada año de trabajo. Art. 196, 197, 198, 199. Cálculo del aguinaldo.",
            "default": "Artículo del Código de Trabajo sobre relación laboral, derechos del trabajador, obligaciones del patrono."
        }
    }

    query_lower = query.lower()

    if tipo_codigo and tipo_codigo in templates_hyde:
        tipo_templates = templates_hyde[tipo_codigo]
        # Buscar template específico para la query
        for key, template in tipo_templates.items():
            if key != "default" and key in query_lower:
                return f"{query}. {template}"
        # Usar template por defecto del tipo
        return f"{query}. {tipo_templates.get('default', '')}"

    return query


def buscar_articulos(query: str, pais: str, top_k: int = 5) -> List[Dict]:
    """
    Busca artículos en Qdrant con filtros inteligentes basados en el tipo de código.

    Mejoras v2.2 (basado en investigación RLM):
    - HyDE: Genera documento hipotético para mejor matching semántico
    - Filtro ESTRICTO: No hace fallback cuando se detecta tipo de código
    - Query enrichment: Añade contexto legal a la búsqueda
    - Fallback inteligente: Solo cuando NO se detecta tipo específico
    """
    coleccion = PAISES[pais]["coleccion"]
    model = get_model()
    client = get_qdrant()

    # Detectar tipo de código legal
    tipo_codigo = detectar_tipo_codigo(query)
    filtro = crear_filtro_codigo(tipo_codigo)

    # Log para debugging
    print(f"🔍 Query: '{query}' → Tipo detectado: {tipo_codigo}")
    print(f"🔍 Filtro: {CODIGO_MAPPING.get(tipo_codigo, 'Sin filtro')}")

    results = []

    if tipo_codigo and filtro:
        # MODO ESTRICTO: Cuando detectamos un tipo específico, FORZAMOS el filtro
        # Usar HyDE para mejorar el matching semántico
        query_hyde = generar_query_hyde(query, tipo_codigo)
        print(f"📝 Query HyDE: '{query_hyde[:100]}...'")

        embedding = model.encode(query_hyde)

        try:
            results = client.search(
                collection_name=coleccion,
                query_vector=embedding.tolist(),
                query_filter=filtro,
                limit=top_k * 2,  # Buscar más para tener margen
                with_payload=True
            )
            print(f"📊 Resultados con filtro {tipo_codigo}: {len(results)}")

            # Si no hay resultados con HyDE, intentar con query enriquecida simple
            if len(results) < top_k:
                query_enriquecida = enriquecer_query(query, tipo_codigo)
                embedding_simple = model.encode(query_enriquecida)

                results_simple = client.search(
                    collection_name=coleccion,
                    query_vector=embedding_simple.tolist(),
                    query_filter=filtro,
                    limit=top_k * 2,
                    with_payload=True
                )
                print(f"📊 Resultados con query enriquecida: {len(results_simple)}")

                # Combinar resultados, evitando duplicados
                seen_ids = {r.id for r in results}
                for r in results_simple:
                    if r.id not in seen_ids:
                        results.append(r)
                        seen_ids.add(r.id)

            # Limitar a top_k
            results = results[:top_k]

        except Exception as e:
            print(f"⚠️ Error en búsqueda con filtro: {e}")
            # Si hay error con el filtro, intentar sin filtro como último recurso
            embedding_original = model.encode(query)
            results = client.search(
                collection_name=coleccion,
                query_vector=embedding_original.tolist(),
                limit=top_k,
                with_payload=True
            )
    else:
        # MODO GENERAL: Sin tipo detectado, búsqueda abierta
        embedding = model.encode(query)
        results = client.search(
            collection_name=coleccion,
            query_vector=embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        print(f"📊 Resultados sin filtro (query general): {len(results)}")

    return [{
        "id": r.payload.get("id", ""),
        "numero": r.payload.get("numero", ""),
        "contenido": r.payload.get("contenido", "")[:1500],
        "codigo": r.payload.get("codigo", ""),
        "score": round(r.score, 4),
        "pais": pais,
        "tipo_detectado": tipo_codigo
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
    
    # Detectar tipo de código para incluir en respuesta
    tipo_codigo = detectar_tipo_codigo(request.query)

    # Buscar artículos (ahora con filtros inteligentes)
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
        tiempo_ms=round((time.time() - inicio) * 1000, 2),
        tipo_codigo_detectado=tipo_codigo
    )

# Endpoint legacy para compatibilidad
@app.post("/api/query")
async def query_legacy(request: ConsultaRequest):
    """Endpoint legacy - redirige a /api/consulta"""
    return await consulta(request)


@app.get("/api/debug/detectar-codigo")
async def debug_detectar_codigo(query: str = Query(..., description="Query a analizar")):
    """
    Endpoint de debug para probar la detección de tipo de código legal.
    Útil para verificar que términos como 'robo', 'hurto', 'legítima defensa'
    son correctamente clasificados como Código Penal.
    """
    tipo_detectado = detectar_tipo_codigo(query)
    query_enriquecida = enriquecer_query(query, tipo_detectado)

    # Mostrar qué términos coincidieron
    query_lower = query.lower()
    terminos_coincidentes = {
        "PENAL": [t for t in TERMINOS_CODIGO_PENAL if t in query_lower],
        "CIVIL": [t for t in TERMINOS_CODIGO_CIVIL if t in query_lower],
        "LABORAL": [t for t in TERMINOS_CODIGO_LABORAL if t in query_lower],
        "PROCESAL": [t for t in TERMINOS_CODIGO_PROCESAL if t in query_lower],
        "CONSTITUCION": [t for t in TERMINOS_CONSTITUCION if t in query_lower],
    }

    return {
        "query_original": query,
        "tipo_codigo_detectado": tipo_detectado,
        "query_enriquecida": query_enriquecida,
        "filtro_qdrant": CODIGO_MAPPING.get(tipo_detectado, []) if tipo_detectado else None,
        "terminos_coincidentes": {k: v for k, v in terminos_coincidentes.items() if v},
        "scores": {
            "PENAL": len(terminos_coincidentes["PENAL"]),
            "CIVIL": len(terminos_coincidentes["CIVIL"]),
            "LABORAL": len(terminos_coincidentes["LABORAL"]),
            "PROCESAL": len(terminos_coincidentes["PROCESAL"]),
            "CONSTITUCION": len(terminos_coincidentes["CONSTITUCION"]),
        }
    }


@app.get("/api/debug/codigos-disponibles")
async def debug_codigos_disponibles(pais: str = Query("SV", description="Código del país")):
    """
    Endpoint de diagnóstico para ver los valores únicos del campo 'codigo' en Qdrant.
    Esto ayuda a identificar los valores exactos para configurar los filtros.
    """
    if pais.upper() not in PAISES:
        raise HTTPException(status_code=400, detail=f"País no soportado: {pais}")

    coleccion = PAISES[pais.upper()]["coleccion"]
    client = get_qdrant()

    # Obtener una muestra de puntos para ver los valores de 'codigo'
    try:
        # Scroll para obtener puntos
        results, _ = client.scroll(
            collection_name=coleccion,
            limit=500,  # Muestra de 500 documentos
            with_payload=True
        )

        # Contar valores únicos del campo 'codigo'
        codigos_count = {}
        for point in results:
            codigo = point.payload.get("codigo", "SIN_CODIGO")
            codigos_count[codigo] = codigos_count.get(codigo, 0) + 1

        # Ordenar por cantidad
        codigos_ordenados = sorted(codigos_count.items(), key=lambda x: -x[1])

        return {
            "pais": pais.upper(),
            "coleccion": coleccion,
            "muestra_total": len(results),
            "codigos_unicos": len(codigos_count),
            "codigos": [{"valor": k, "cantidad": v} for k, v in codigos_ordenados],
            "nota": "Usar estos valores exactos en CODIGO_MAPPING para que los filtros funcionen"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Qdrant: {str(e)}")


@app.get("/api/debug/contar-todos")
async def debug_contar_todos(pais: str = Query("SV", description="Código del país")):
    """
    Cuenta TODOS los artículos por código en la colección completa.
    Usa scroll iterativo para obtener el conteo exacto.
    """
    if pais.upper() not in PAISES:
        raise HTTPException(status_code=400, detail=f"País no soportado: {pais}")

    coleccion = PAISES[pais.upper()]["coleccion"]
    client = get_qdrant()

    try:
        # Scroll completo para contar por código
        codigos_count = {}
        offset = None
        total_procesados = 0

        while True:
            results, offset = client.scroll(
                collection_name=coleccion,
                limit=1000,
                offset=offset,
                with_payload=["codigo"]  # Solo traer el campo codigo para ser eficiente
            )

            if not results:
                break

            for point in results:
                codigo = point.payload.get("codigo", "SIN_CODIGO")
                codigos_count[codigo] = codigos_count.get(codigo, 0) + 1
                total_procesados += 1

            if offset is None:
                break

        # Ordenar por cantidad
        codigos_ordenados = sorted(codigos_count.items(), key=lambda x: -x[1])

        return {
            "pais": pais.upper(),
            "coleccion": coleccion,
            "total_articulos": total_procesados,
            "codigos_unicos": len(codigos_count),
            "codigos": [{"valor": k, "cantidad": v, "porcentaje": round(v/total_procesados*100, 2) if total_procesados > 0 else 0} for k, v in codigos_ordenados],
            "resumen_principales": {
                "Codigo Penal": codigos_count.get("Codigo Penal", 0),
                "Codigo Civil": codigos_count.get("Codigo Civil", 0),
                "Codigo De Trabajo": codigos_count.get("Codigo De Trabajo", 0),
                "Constitucion": codigos_count.get("Constitucion", 0),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Qdrant: {str(e)}")

@app.get("/api/debug/test-filtro")
async def debug_test_filtro(
    query: str = Query("robo", description="Query a buscar"),
    pais: str = Query("SV", description="Código del país")
):
    """
    Debug: Prueba directamente el filtro de Qdrant para verificar que funciona.
    """
    if pais.upper() not in PAISES:
        raise HTTPException(status_code=400, detail=f"País no soportado: {pais}")

    coleccion = PAISES[pais.upper()]["coleccion"]
    model = get_model()
    client = get_qdrant()

    tipo_codigo = detectar_tipo_codigo(query)
    valores_filtro = CODIGO_MAPPING.get(tipo_codigo, [])

    # Crear embedding simple
    embedding = model.encode(query)

    # 1. Búsqueda SIN filtro
    sin_filtro = client.search(
        collection_name=coleccion,
        query_vector=embedding.tolist(),
        limit=5,
        with_payload=True
    )

    # 2. Búsqueda CON filtro (si hay tipo detectado)
    con_filtro = []
    filtro_usado = None
    if tipo_codigo and valores_filtro:
        filtro_usado = {
            "must": [{"key": "codigo", "match": {"any": valores_filtro}}]
        }
        try:
            filtro = Filter(
                must=[
                    FieldCondition(
                        key="codigo",
                        match=MatchAny(any=valores_filtro)
                    )
                ]
            )
            con_filtro = client.search(
                collection_name=coleccion,
                query_vector=embedding.tolist(),
                query_filter=filtro,
                limit=5,
                with_payload=True
            )
        except Exception as e:
            return {"error": f"Error con filtro: {str(e)}"}

    return {
        "query": query,
        "tipo_detectado": tipo_codigo,
        "valores_filtro": valores_filtro,
        "filtro_usado": filtro_usado,
        "resultados_SIN_filtro": [
            {"numero": r.payload.get("numero"), "codigo": r.payload.get("codigo"), "score": round(r.score, 4)}
            for r in sin_filtro
        ],
        "resultados_CON_filtro": [
            {"numero": r.payload.get("numero"), "codigo": r.payload.get("codigo"), "score": round(r.score, 4)}
            for r in con_filtro
        ],
        "cantidad_sin_filtro": len(sin_filtro),
        "cantidad_con_filtro": len(con_filtro)
    }


@app.post("/api/admin/crear-indice-codigo")
async def crear_indice_codigo(pais: str = Query("SV", description="Código del país")):
    """
    ADMIN: Crea el índice necesario para el campo 'codigo' en Qdrant.
    Esto es necesario para que los filtros funcionen.
    """
    from qdrant_client.models import PayloadSchemaType

    if pais.upper() not in PAISES:
        raise HTTPException(status_code=400, detail=f"País no soportado: {pais}")

    coleccion = PAISES[pais.upper()]["coleccion"]
    client = get_qdrant()

    try:
        # Crear índice de tipo keyword para el campo 'codigo'
        client.create_payload_index(
            collection_name=coleccion,
            field_name="codigo",
            field_schema=PayloadSchemaType.KEYWORD
        )
        return {
            "success": True,
            "message": f"Índice creado para campo 'codigo' en colección '{coleccion}'",
            "pais": pais.upper()
        }
    except Exception as e:
        # Si ya existe, no es error
        if "already exists" in str(e).lower():
            return {
                "success": True,
                "message": f"Índice ya existía para campo 'codigo' en colección '{coleccion}'",
                "pais": pais.upper()
            }
        raise HTTPException(status_code=500, detail=f"Error creando índice: {str(e)}")


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

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS DE FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

# Almacenamiento temporal de feedback (para producción usar base de datos)
_feedback_storage = []

class FeedbackRequest(BaseModel):
    query: str
    response_id: Optional[str] = None
    rating: str  # "positive", "negative", "partial"
    comment: Optional[str] = None
    expected_code: Optional[str] = None  # Código esperado (PENAL, CIVIL, etc.)
    received_code: Optional[str] = None  # Código recibido
    articles_returned: Optional[List[str]] = None  # Lista de artículos devueltos
    pais: str = "SV"

class FeedbackResponse(BaseModel):
    success: bool
    feedback_id: str
    message: str

@app.post("/api/feedback", response_model=FeedbackResponse)
async def enviar_feedback(
    feedback: FeedbackRequest,
    x_user_id: Optional[str] = Header(None)
):
    """
    Recibe feedback sobre la calidad de las respuestas.
    Útil para mejorar el sistema de RAG.
    """
    import uuid

    feedback_id = str(uuid.uuid4())
    user_id = x_user_id or "anonymous"

    feedback_entry = {
        "id": feedback_id,
        "user_id": user_id,
        "query": feedback.query,
        "response_id": feedback.response_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "expected_code": feedback.expected_code,
        "received_code": feedback.received_code,
        "articles_returned": feedback.articles_returned,
        "pais": feedback.pais,
        "created_at": datetime.now().isoformat(),
        "version": VERSION
    }

    _feedback_storage.append(feedback_entry)

    # Log para análisis
    rating_emoji = {"positive": "👍", "negative": "👎", "partial": "🔄"}.get(feedback.rating, "❓")
    print(f"📝 Feedback {rating_emoji}: '{feedback.query}' - {feedback.comment or 'Sin comentario'}")

    return FeedbackResponse(
        success=True,
        feedback_id=feedback_id,
        message="Feedback recibido correctamente. ¡Gracias por ayudarnos a mejorar!"
    )

@app.get("/api/feedback")
async def listar_feedback(
    limit: int = Query(50, description="Número máximo de registros"),
    rating: Optional[str] = Query(None, description="Filtrar por rating: positive, negative, partial")
):
    """
    Lista el feedback recibido (para análisis).
    """
    feedback_list = _feedback_storage.copy()

    # Filtrar por rating si se especifica
    if rating:
        feedback_list = [f for f in feedback_list if f.get("rating") == rating]

    # Ordenar por fecha (más reciente primero) y limitar
    feedback_list = sorted(feedback_list, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    # Estadísticas
    total = len(_feedback_storage)
    stats = {
        "total": total,
        "positive": len([f for f in _feedback_storage if f.get("rating") == "positive"]),
        "negative": len([f for f in _feedback_storage if f.get("rating") == "negative"]),
        "partial": len([f for f in _feedback_storage if f.get("rating") == "partial"]),
    }

    return {
        "feedback": feedback_list,
        "stats": stats,
        "success_rate": round(stats["positive"] / total * 100, 2) if total > 0 else 0
    }

@app.get("/api/feedback/stats")
async def estadisticas_feedback():
    """
    Estadísticas resumidas del feedback.
    """
    total = len(_feedback_storage)

    if total == 0:
        return {
            "total": 0,
            "message": "No hay feedback registrado aún"
        }

    # Contar por rating
    by_rating = {
        "positive": len([f for f in _feedback_storage if f.get("rating") == "positive"]),
        "negative": len([f for f in _feedback_storage if f.get("rating") == "negative"]),
        "partial": len([f for f in _feedback_storage if f.get("rating") == "partial"]),
    }

    # Contar por código esperado vs recibido (para detectar problemas)
    mismatches = [f for f in _feedback_storage
                  if f.get("expected_code") and f.get("received_code")
                  and f.get("expected_code") != f.get("received_code")]

    # Queries con feedback negativo (para priorizar mejoras)
    problematic_queries = [
        {"query": f.get("query"), "comment": f.get("comment"), "expected": f.get("expected_code")}
        for f in _feedback_storage if f.get("rating") == "negative"
    ]

    return {
        "total_feedback": total,
        "by_rating": by_rating,
        "success_rate": round(by_rating["positive"] / total * 100, 2),
        "partial_rate": round(by_rating["partial"] / total * 100, 2),
        "error_rate": round(by_rating["negative"] / total * 100, 2),
        "code_mismatches": len(mismatches),
        "problematic_queries": problematic_queries[:10],  # Top 10 queries problemáticas
        "version": VERSION
    }
