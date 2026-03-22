import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

class RootErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error("Root render failure", error);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          background: "#0b1016",
          color: "#ebf1f5",
          fontFamily: "Segoe UI, sans-serif",
          padding: "24px"
        }}>
          <div style={{
            width: "min(840px, 100%)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "16px",
            background: "rgba(255,255,255,0.03)",
            padding: "20px"
          }}>
            <h1 style={{ marginTop: 0 }}>Dashboard runtime error</h1>
            <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
              {String(this.state.error?.message || this.state.error)}
            </pre>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>
);
