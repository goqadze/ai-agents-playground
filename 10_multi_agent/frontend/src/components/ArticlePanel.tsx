interface Props {
  topic: string
  article: string
}

/**
 * Renders the final article as formatted HTML.
 *
 * We do a lightweight markdown conversion instead of pulling in react-markdown,
 * covering the subset the Writer agent actually produces:
 *   ## heading → <h2>
 *   **bold**   → <strong>
 *   blank line → paragraph break
 */
function renderMarkdown(md: string): string {
  return md
    // headings
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // paragraph breaks (two+ newlines between text)
    .split(/\n{2,}/)
    .map(block => {
      const trimmed = block.trim()
      if (!trimmed) return ''
      if (trimmed.startsWith('<h')) return trimmed
      return `<p>${trimmed.replace(/\n/g, ' ')}</p>`
    })
    .join('\n')
}

export default function ArticlePanel({ topic, article }: Props) {
  if (!article) return null

  return (
    <div className="article-panel">
      <div className="article-panel__header">
        <span className="article-icon">📄</span>
        <div>
          <div className="article-title">Final Article</div>
          <div className="article-subtitle">{topic}</div>
        </div>
        <button
          className="copy-btn"
          onClick={() => navigator.clipboard.writeText(article)}
          title="Copy markdown to clipboard"
        >
          Copy
        </button>
      </div>

      <div
        className="article-content"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(article) }}
      />
    </div>
  )
}
