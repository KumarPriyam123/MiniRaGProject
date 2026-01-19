import { useState, useRef } from 'react'

const API_BASE = '/api'

// Rough token estimate: ~4 chars per token for English
const estimateTokens = (text) => Math.ceil((text?.length || 0) / 4)

export default function App() {
  // Ingest state
  const [ingestText, setIngestText] = useState('')
  const [docTitle, setDocTitle] = useState('')
  const [ingestStatus, setIngestStatus] = useState(null)
  const [ingesting, setIngesting] = useState(false)

  // File upload state
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const fileInputRef = useRef(null)

  // Query state
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [querying, setQuerying] = useState(false)
  const [latency, setLatency] = useState(null)

  // Ingest handler
  const handleIngest = async () => {
    if (!ingestText.trim()) return
    
    setIngesting(true)
    setIngestStatus(null)
    
    try {
      const res = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: ingestText,
          metadata: { title: docTitle || 'Untitled' }
        })
      })
      const data = await res.json()
      setIngestStatus({
        success: data.status === 'success',
        message: `Created ${data.chunks_created} chunks (doc: ${data.doc_id})`
      })
      if (data.status === 'success') {
        setIngestText('')
        setDocTitle('')
      }
    } catch (err) {
      setIngestStatus({ success: false, message: err.message })
    } finally {
      setIngesting(false)
    }
  }

  // File upload handler
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setUploadStatus(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      })

      const data = await res.json()

      if (!res.ok) {
        setUploadStatus({
          success: false,
          message: data.detail || 'Upload failed'
        })
      } else {
        setUploadStatus({
          success: data.status === 'success',
          message: data.message || `Created ${data.chunks_created} chunks from ${data.filename}`
        })
      }
    } catch (err) {
      setUploadStatus({ success: false, message: err.message })
    } finally {
      setUploading(false)
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  // Query handler
  const handleQuery = async () => {
    if (!query.trim()) return
    
    setQuerying(true)
    setResult(null)
    setLatency(null)
    
    const start = performance.now()
    
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query })
      })
      const data = await res.json()
      setLatency(Math.round(performance.now() - start))
      setResult(data)
    } catch (err) {
      setResult({ answer: `Error: ${err.message}`, sources: [], has_answer: false })
      setLatency(Math.round(performance.now() - start))
    } finally {
      setQuerying(false)
    }
  }

  // Render answer with highlighted citations
  const renderAnswer = (answer) => {
    if (!answer) return null
    
    // Replace [1], [2] etc with styled spans
    const parts = answer.split(/(\[\d+\])/)
    return parts.map((part, i) => {
      const match = part.match(/^\[(\d+)\]$/)
      if (match) {
        return (
          <a
            key={i}
            href={`#source-${match[1]}`}
            className="citation"
            title={`Jump to source ${match[1]}`}
          >
            {part}
          </a>
        )
      }
      return <span key={i}>{part}</span>
    })
  }

  return (
    <div className="container">
      <h1>🔍 Mini RAG</h1>
      
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* INGEST SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section className="card">
        <h2>📄 Ingest Document</h2>
        
        <input
          type="text"
          placeholder="Document title (optional)"
          value={docTitle}
          onChange={(e) => setDocTitle(e.target.value)}
          className="input"
        />
        
        <textarea
          placeholder="Paste your text here..."
          value={ingestText}
          onChange={(e) => setIngestText(e.target.value)}
          rows={8}
          className="textarea"
        />
        
        <div className="row">
          <span className="meta">
            ~{estimateTokens(ingestText).toLocaleString()} tokens
          </span>
          <button 
            onClick={handleIngest} 
            disabled={ingesting || !ingestText.trim()}
            className="btn"
          >
            {ingesting ? 'Ingesting...' : 'Ingest'}
          </button>
        </div>
        
        {ingestStatus && (
          <div className={`status ${ingestStatus.success ? 'success' : 'error'}`}>
            {ingestStatus.message}
          </div>
        )}

        {/* File Upload Divider */}
        <div className="divider">
          <span>OR upload a file</span>
        </div>

        {/* File Upload */}
        <div className="upload-section">
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf,.docx"
            onChange={handleFileUpload}
            disabled={uploading}
            className="file-input"
            id="file-upload"
          />
          <label htmlFor="file-upload" className={`file-label ${uploading ? 'disabled' : ''}`}>
            {uploading ? '📤 Uploading...' : '📁 Choose file (.txt, .pdf, .docx)'}
          </label>
          <span className="meta">Max 10MB</span>
        </div>

        {uploadStatus && (
          <div className={`status ${uploadStatus.success ? 'success' : 'error'}`}>
            {uploadStatus.message}
          </div>
        )}
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* QUERY SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section className="card">
        <h2>💬 Ask a Question</h2>
        
        <div className="query-row">
          <input
            type="text"
            placeholder="What would you like to know?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
            className="input query-input"
          />
          <button 
            onClick={handleQuery} 
            disabled={querying || !query.trim()}
            className="btn"
          >
            {querying ? 'Searching...' : 'Ask'}
          </button>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* RESULTS SECTION */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {result && (
        <section className="card results">
          <h2>📝 Answer</h2>
          
          {/* Metrics bar */}
          <div className="metrics">
            <span>⏱️ {latency}ms</span>
            <span>🎯 {result.tokens_used || 0} tokens</span>
            <span>📚 {result.sources?.length || 0} sources</span>
            {!result.has_answer && <span className="warning">⚠️ Low confidence</span>}
          </div>
          
          {/* Answer text with citation links */}
          <div className="answer">
            {renderAnswer(result.answer)}
          </div>
          
          {/* Sources list */}
          {result.sources?.length > 0 && (
            <div className="sources">
              <h3>Sources</h3>
              {result.sources.map((src, i) => (
                <div key={src.chunk_id} id={`source-${i + 1}`} className="source">
                  <div className="source-header">
                    <span className="source-index">[{i + 1}]</span>
                    <span className="source-title">{src.title}</span>
                    <span className="source-score">
                      {(src.score * 100).toFixed(1)}% match
                    </span>
                  </div>
                  <p className="source-text">{src.text}</p>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
