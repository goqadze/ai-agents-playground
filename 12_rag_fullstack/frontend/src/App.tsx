import Chat from "./components/Chat";
import Ingest from "./components/Ingest";

export default function App() {
  return (
    <div className="app-layout">
      <header className="app-header">
        <h1>RAG Fullstack</h1>
        <span className="app-subtitle">
          LangChain · LangGraph · pgvector · @langchain/react
        </span>
      </header>
      <div className="app-body">
        <Ingest />
        <Chat />
      </div>
    </div>
  );
}
