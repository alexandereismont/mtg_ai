import { useState, useRef, useEffect } from 'react'
import './App.css'

const WELCOME = {
  id: 'welcome',
  role: 'assistant',
  content:
    "Hello! I'm your MTG Rules Judge. Ask me any question about Magic: The Gathering rules and I'll find the relevant rules and explain them.",
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      {!isUser && (
        <div className="avatar assistant-avatar" aria-hidden="true">
          ⚖
        </div>
      )}
      <div className={`bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        {msg.content}
        {msg.streaming && <span className="cursor" aria-hidden="true" />}
      </div>
      {isUser && (
        <div className="avatar user-avatar" aria-hidden="true">
          🧙
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(question) {
    const userMsg = { id: Date.now(), role: 'user', content: question }
    const assistantId = Date.now() + 1
    const assistantMsg = { id: assistantId, role: 'assistant', content: '', streaming: true }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setLoading(true)

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6)
          if (payload === '[DONE]') break
          try {
            const { token } = JSON.parse(payload)
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: m.content + token } : m,
              ),
            )
          } catch {
            // malformed chunk — ignore
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Error: ${err.message}` }
            : m,
        ),
      )
    } finally {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)),
      )
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    sendMessage(q)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  function handleInputChange(e) {
    setInput(e.target.value)
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }

  const suggestions = [
    'What is first strike?',
    'How does deathtouch work?',
    'Can a creature with summoning sickness block?',
    'What happens during the upkeep step?',
  ]

  return (
    <div className="app">
      <header className="header">
        <span className="header-icon" aria-hidden="true">⚔</span>
        <h1 className="header-title">MTG Rules Judge</h1>
        <span className="header-sub">Powered by Ollama + TF-IDF RAG</span>
      </header>

      <main className="messages" aria-live="polite">
        {messages.map((msg) => (
          <Message key={msg.id} msg={msg} />
        ))}

        {messages.length === 1 && (
          <div className="suggestions">
            {suggestions.map((s) => (
              <button
                key={s}
                className="suggestion-chip"
                onClick={() => {
                  setInput(s)
                  inputRef.current?.focus()
                }}
              >
                {s}
              </button>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <form className="input-bar" onSubmit={handleSubmit}>
        <textarea
          ref={(el) => {
            textareaRef.current = el
            inputRef.current = el
          }}
          className="input-field"
          rows={1}
          placeholder="Ask a rules question… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          disabled={loading}
          aria-label="Rules question"
        />
        <button
          type="submit"
          className="send-btn"
          disabled={loading || !input.trim()}
          aria-label="Send"
        >
          {loading ? (
            <span className="spinner" aria-hidden="true" />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </form>
    </div>
  )
}
