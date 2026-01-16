import React, { useState, useRef, useEffect } from 'react'
import {
  Send,
  MessageSquare,
  Plus,
  Trash2,
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  Scale,
  ChevronDown,
  BarChart3,
  Menu,
  X,
  Loader2,
  BookOpen,
  AlertCircle
} from 'lucide-react'

// API Configuration
const API_URL = 'https://api-graphrag-v21.bravecliff-83b394ec.eastus.azurecontainerapps.io'

// Countries configuration
const PAISES = {
  SV: { nombre: 'El Salvador', bandera: '🇸🇻' },
  GT: { nombre: 'Guatemala', bandera: '🇬🇹' },
  CR: { nombre: 'Costa Rica', bandera: '🇨🇷' },
  PA: { nombre: 'Panamá', bandera: '🇵🇦' },
  MX: { nombre: 'México', bandera: '🇲🇽' },
}

// Complexity badge colors
const COMPLEXITY_STYLES = {
  simple: 'bg-green-100 text-green-800',
  medium: 'bg-blue-100 text-blue-800',
  complex: 'bg-purple-100 text-purple-800',
}

const COMPLEXITY_LABELS = {
  simple: 'Búsqueda rápida',
  medium: 'Búsqueda optimizada',
  complex: 'Análisis profundo',
}

function App() {
  const [conversations, setConversations] = useState([])
  const [currentConversation, setCurrentConversation] = useState(null)
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedCountry, setSelectedCountry] = useState('SV')
  const [showSidebar, setShowSidebar] = useState(true)
  const [showStats, setShowStats] = useState(false)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Create new conversation
  const createNewConversation = () => {
    const newConv = {
      id: Date.now().toString(),
      title: 'Nueva consulta',
      messages: [],
      createdAt: new Date().toISOString(),
      country: selectedCountry
    }
    setConversations(prev => [newConv, ...prev])
    setCurrentConversation(newConv.id)
    setMessages([])
    setError(null)
  }

  // Delete conversation
  const deleteConversation = (id) => {
    setConversations(prev => prev.filter(c => c.id !== id))
    if (currentConversation === id) {
      setCurrentConversation(null)
      setMessages([])
    }
  }

  // Load conversation
  const loadConversation = (conv) => {
    setCurrentConversation(conv.id)
    setMessages(conv.messages || [])
    setSelectedCountry(conv.country || 'SV')
  }

  // Send query to API
  const sendQuery = async (query) => {
    if (!query.trim()) return

    setError(null)

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/api/consulta`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          pais: selectedCountry,
          top_k: 5,
          generar_respuesta: true
        })
      })

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()

      // Add assistant message
      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.respuesta || 'No se pudo generar una respuesta.',
        articulos: data.articulos,
        metadata: {
          tiempo_ms: data.tiempo_ms,
          tipo_codigo: data.tipo_codigo_detectado,
          complejidad: data.complejidad_query,
          reranking: data.reranking_aplicado
        },
        timestamp: new Date().toISOString()
      }

      setMessages(prev => [...prev, assistantMessage])

      // Update conversation
      if (currentConversation) {
        setConversations(prev => prev.map(c =>
          c.id === currentConversation
            ? { ...c, messages: [...(c.messages || []), userMessage, assistantMessage], title: query.slice(0, 50) }
            : c
        ))
      }

    } catch (err) {
      console.error('Error:', err)
      setError(err.message)

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: 'error',
        content: `Error al procesar la consulta: ${err.message}`,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  // Send feedback
  const sendFeedback = async (messageId, rating, isRelevant) => {
    try {
      const message = messages.find(m => m.id === messageId)
      if (!message) return

      await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: messages.find(m => m.role === 'user')?.content || '',
          rating: rating,
          es_relevante: isRelevant,
          pais: selectedCountry
        })
      })

      // Update message to show feedback was sent
      setMessages(prev => prev.map(m =>
        m.id === messageId ? { ...m, feedbackSent: true, feedbackRating: rating } : m
      ))
    } catch (err) {
      console.error('Error sending feedback:', err)
    }
  }

  // Load stats
  const loadStats = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stats`)
      const data = await response.json()
      setStats(data)
      setShowStats(true)
    } catch (err) {
      console.error('Error loading stats:', err)
    }
  }

  // Handle form submit
  const handleSubmit = (e) => {
    e.preventDefault()
    if (!currentConversation) {
      createNewConversation()
    }
    sendQuery(inputValue)
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className={`${showSidebar ? 'w-72' : 'w-0'} bg-white border-r border-gray-200 flex flex-col transition-all duration-300 overflow-hidden`}>
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <Scale className="w-8 h-8 text-primary-600" />
            <div>
              <h1 className="font-bold text-lg text-gray-900">RAG Legal</h1>
              <p className="text-xs text-gray-500">v2.4.0</p>
            </div>
          </div>

          <button
            onClick={createNewConversation}
            className="w-full flex items-center justify-center gap-2 bg-primary-600 text-white py-2.5 px-4 rounded-lg hover:bg-primary-700 transition-colors"
          >
            <Plus className="w-5 h-5" />
            Nueva Consulta
          </button>
        </div>

        {/* Conversations list */}
        <div className="flex-1 overflow-y-auto p-2">
          <p className="text-xs font-medium text-gray-400 px-2 py-1">CONVERSACIONES</p>
          {conversations.length === 0 ? (
            <p className="text-sm text-gray-400 px-2 py-4 text-center">
              No hay consultas aún
            </p>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                onClick={() => loadConversation(conv)}
                className={`group flex items-center gap-2 p-2.5 rounded-lg cursor-pointer mb-1 ${
                  currentConversation === conv.id
                    ? 'bg-primary-50 text-primary-700'
                    : 'hover:bg-gray-100'
                }`}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1 text-sm truncate">{conv.title}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id) }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded"
                >
                  <Trash2 className="w-3.5 h-3.5 text-red-500" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Stats button */}
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={loadStats}
            className="w-full flex items-center justify-center gap-2 text-gray-600 py-2 px-4 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <BarChart3 className="w-4 h-4" />
            Estadísticas
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-2 hover:bg-gray-100 rounded-lg lg:hidden"
            >
              {showSidebar ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>

            {/* Country selector */}
            <div className="relative">
              <select
                value={selectedCountry}
                onChange={(e) => setSelectedCountry(e.target.value)}
                className="appearance-none bg-gray-100 border-0 rounded-lg py-2 pl-3 pr-10 text-sm font-medium cursor-pointer hover:bg-gray-200 transition-colors"
              >
                {Object.entries(PAISES).map(([code, { nombre, bandera }]) => (
                  <option key={code} value={code}>
                    {bandera} {nombre}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">
              RAG Legal {PAISES[selectedCountry].bandera}
            </span>
          </div>
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && !currentConversation && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Scale className="w-16 h-16 text-primary-200 mb-4" />
              <h2 className="text-xl font-semibold text-gray-700 mb-2">
                Bienvenido a RAG Legal
              </h2>
              <p className="text-gray-500 max-w-md mb-6">
                Realiza consultas legales sobre códigos y leyes de {PAISES[selectedCountry].nombre}.
                El sistema utiliza IA avanzada para encontrar los artículos más relevantes.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg">
                {[
                  '¿Cuál es la indemnización por despido?',
                  '¿Qué dice la ley sobre el aguinaldo?',
                  '¿Cuáles son las causas de divorcio?',
                  '¿Qué es la legítima defensa?'
                ].map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => { createNewConversation(); sendQuery(suggestion) }}
                    className="text-left p-3 bg-white border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors text-sm"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onFeedback={sendFeedback}
            />
          ))}

          {isLoading && (
            <div className="flex items-center gap-2 text-gray-500 animate-pulse-slow">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Buscando en la base de datos legal...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="bg-white border-t border-gray-200 p-4">
          {error && (
            <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700 text-sm">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Escribe tu consulta legal..."
              disabled={isLoading}
              className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100"
            />
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="bg-primary-600 text-white px-5 py-3 rounded-xl hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <p className="text-xs text-gray-400 mt-2 text-center">
            Presiona Enter para enviar • Shift+Enter para nueva línea
          </p>
        </div>
      </main>

      {/* Stats modal */}
      {showStats && stats && (
        <StatsModal stats={stats} onClose={() => setShowStats(false)} />
      )}
    </div>
  )
}

// Message bubble component
function MessageBubble({ message, onFeedback }) {
  const isUser = message.role === 'user'
  const isError = message.role === 'error'

  if (isError) {
    return (
      <div className="flex justify-center animate-fade-in">
        <div className="bg-red-50 text-red-700 rounded-lg px-4 py-2 max-w-lg text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}>
      <div className={`max-w-2xl ${isUser ? 'message-user px-4 py-3' : ''}`}>
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="space-y-3">
            {/* Metadata badges */}
            {message.metadata && (
              <div className="flex flex-wrap items-center gap-2 mb-2">
                {message.metadata.reranking && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded-full text-xs">
                    <Sparkles className="w-3 h-3" />
                    IA Optimizada
                  </span>
                )}
                {message.metadata.complejidad && (
                  <span className={`px-2 py-0.5 rounded-full text-xs ${COMPLEXITY_STYLES[message.metadata.complejidad]}`}>
                    {COMPLEXITY_LABELS[message.metadata.complejidad]}
                  </span>
                )}
                {message.metadata.tipo_codigo && (
                  <span className="px-2 py-0.5 bg-gray-100 text-gray-700 rounded-full text-xs">
                    {message.metadata.tipo_codigo}
                  </span>
                )}
                {message.metadata.tiempo_ms && (
                  <span className="text-xs text-gray-400">
                    {(message.metadata.tiempo_ms / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
            )}

            {/* Response content */}
            <div className="message-assistant px-4 py-3">
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>

            {/* Articles */}
            {message.articulos && message.articulos.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-gray-500 flex items-center gap-1">
                  <BookOpen className="w-3 h-3" />
                  Artículos relevantes ({message.articulos.length})
                </p>
                {message.articulos.slice(0, 3).map((art, i) => (
                  <div key={i} className="bg-white border border-gray-200 rounded-lg p-3 text-sm">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-primary-700">{art.numero}</span>
                      <span className="text-xs text-gray-400">{art.codigo}</span>
                    </div>
                    <p className="text-gray-600 text-xs line-clamp-3">{art.contenido}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-gray-400">
                        Score: {(art.score * 100).toFixed(0)}%
                      </span>
                      {art.reranked && art.original_score && (
                        <span className="text-xs text-green-600">
                          (+{((art.score - art.original_score) * 100).toFixed(0)}% rerank)
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Feedback buttons */}
            {!message.feedbackSent ? (
              <div className="flex items-center gap-2 pt-2">
                <span className="text-xs text-gray-400">¿Te fue útil?</span>
                <button
                  onClick={() => onFeedback(message.id, 5, true)}
                  className="p-1.5 hover:bg-green-100 rounded-lg transition-colors"
                  title="Sí, útil"
                >
                  <ThumbsUp className="w-4 h-4 text-gray-400 hover:text-green-600" />
                </button>
                <button
                  onClick={() => onFeedback(message.id, 1, false)}
                  className="p-1.5 hover:bg-red-100 rounded-lg transition-colors"
                  title="No, mejorable"
                >
                  <ThumbsDown className="w-4 h-4 text-gray-400 hover:text-red-600" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-1 pt-2 text-xs text-gray-400">
                <span>Gracias por tu feedback</span>
                {message.feedbackRating === 5 ? (
                  <ThumbsUp className="w-3 h-3 text-green-500" />
                ) : (
                  <ThumbsDown className="w-3 h-3 text-red-500" />
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// Stats modal component
function StatsModal({ stats, onClose }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-white rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Estadísticas del Sistema</h2>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-primary-50 rounded-lg p-3">
              <p className="text-2xl font-bold text-primary-700">{stats.total_vectores?.toLocaleString() || 0}</p>
              <p className="text-xs text-primary-600">Artículos totales</p>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <p className="text-2xl font-bold text-green-700">{stats.paises_activos || 0}</p>
              <p className="text-xs text-green-600">Países activos</p>
            </div>
          </div>

          {stats.por_pais && (
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Por país</p>
              <div className="space-y-2">
                {Object.entries(stats.por_pais).map(([code, data]) => (
                  <div key={code} className="flex items-center justify-between bg-gray-50 rounded-lg p-2">
                    <span className="text-sm">
                      {PAISES[code]?.bandera} {data.nombre}
                    </span>
                    <span className="text-sm font-medium">{data.vectores?.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-center text-xs text-gray-400 pt-2">
            Versión {stats.version || '2.4.0'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
